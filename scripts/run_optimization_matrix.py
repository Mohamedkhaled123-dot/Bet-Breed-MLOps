import copy
import json
from pathlib import Path
import platform
import time

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
import tensorrt as trt
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.ao.quantization import get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx

from bet_breed_mlops.calibration import calculate_ece
from bet_breed_mlops.data import (
    MANIFEST_PATH,
    PetBreedDataset,
    create_dataloaders,
    eval_transform,
)
from bet_breed_mlops.model import PetBreedClassifier
from bet_breed_mlops.optimization import (
    apply_structured_filter_pruning,
    distillation_loss,
    select_stratified_indices,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CHECKPOINT = PROJECT_ROOT / "artifacts" / "checkpoints" / "resnet18.pt"
TEACHER_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "experiments"
    / "checkpoints"
    / "resnet50_lr_0.001.pt"
)
INITIAL_STUDENT_CHECKPOINT = (
    PROJECT_ROOT / "artifacts" / "checkpoints" / "mobilenet_v3_small.pt"
)
FLOAT_ONNX = PROJECT_ROOT / "artifacts" / "models" / "resnet18.onnx"
INT8_ONNX = PROJECT_ROOT / "artifacts" / "models" / "model_int8.onnx"
TENSORRT_ENGINE = (
    PROJECT_ROOT / "artifacts" / "tensorrt" / "resnet18_fp16.engine"
)
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "optimization"
REPORT_PATH = PROJECT_ROOT / "reports" / "optimization_benchmark.json"
NUM_CLASSES = 37
BATCH_SIZE = 32
QAT_EPOCHS = 2
DISTILL_EPOCHS = 3
DISTILLATION_TEMPERATURE = 3.0
DISTILLATION_WEIGHT = 0.3
CALIBRATION_SAMPLES_PER_CLASS = 4
LATENCY_WARMUP = 20
LATENCY_ITERATIONS = 100
PRUNING_ACCURACY_DROP = 0.01
PRUNING_AMOUNTS = (0.02, 0.05, 0.10, 0.15, 0.30)
INPUT_NAME = "image"
OUTPUT_NAME = "logits"
INPUT_SHAPE = (1, 3, 224, 224)


def load_resnet18(device: torch.device) -> PetBreedClassifier:
    checkpoint = torch.load(
        PRODUCTION_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    model = PetBreedClassifier(
        num_classes=NUM_CLASSES,
        backbone_name="resnet18",
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def load_supervised_student(device: torch.device) -> PetBreedClassifier:
    checkpoint = torch.load(
        INITIAL_STUDENT_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    model = PetBreedClassifier(
        num_classes=NUM_CLASSES,
        backbone_name="mobilenet_v3_small",
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    predictions = logits.argmax(dim=1)
    return {
        "top1": float(predictions.eq(labels).float().mean().item()),
        "macro_f1": float(
            f1_score(
                labels.numpy(),
                predictions.numpy(),
                labels=list(range(logits.shape[1])),
                average="macro",
                zero_division=0,
            )
        ),
        "ece": calculate_ece(logits, labels),
        "examples": int(labels.shape[0]),
    }


def evaluate_torch_model(
    model: nn.Module,
    loader,
    device: torch.device,
) -> dict:
    model.eval()
    logits_batches = []
    label_batches = []
    with torch.inference_mode():
        for images, labels in loader:
            logits_batches.append(
                model(images.to(device, non_blocking=True)).float().cpu()
            )
            label_batches.append(labels.cpu())
    return classification_metrics(
        torch.cat(logits_batches),
        torch.cat(label_batches),
    )


class PetCalibrationReader(CalibrationDataReader):
    def __init__(
        self,
        dataset: Dataset,
        indices: list[int],
        input_name: str,
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.input_name = input_name
        self.rewind()

    def get_next(self):
        try:
            index = next(self._indices)
        except StopIteration:
            return None
        image, _ = self.dataset[index]
        return {
            self.input_name: image.unsqueeze(0)
            .numpy()
            .astype(np.float32, copy=False)
        }

    def rewind(self) -> None:
        self._indices = iter(self.indices)


def quantize_onnx() -> tuple[ort.InferenceSession, int]:
    if not FLOAT_ONNX.exists():
        raise FileNotFoundError(FLOAT_ONNX)
    dataset = PetBreedDataset(
        manifest_path=MANIFEST_PATH,
        split="train",
        transform=eval_transform,
    )
    labels = [record["class_index"] for record in dataset.records]
    indices = select_stratified_indices(
        labels,
        samples_per_class=CALIBRATION_SAMPLES_PER_CLASS,
        seed=20261008,
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    float_session = ort.InferenceSession(
        str(FLOAT_ONNX),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    reader = PetCalibrationReader(
        dataset,
        indices,
        float_session.get_inputs()[0].name,
    )
    INT8_ONNX.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(FLOAT_ONNX),
        model_output=str(INT8_ONNX),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
        },
    )
    session = ort.InferenceSession(
        str(INT8_ONNX),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    print(
        f"Static INT8 PTQ calibrated from {len(indices)} stratified "
        f"training examples."
    )
    return session, len(indices)


def evaluate_onnx(session: ort.InferenceSession, loader) -> dict:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logits_batches = []
    label_batches = []
    for images, labels in loader:
        logits = session.run(
            [output_name],
            {input_name: images.numpy().astype(np.float32, copy=False)},
        )[0]
        logits_batches.append(torch.from_numpy(logits).float())
        label_batches.append(labels.cpu())
    return classification_metrics(
        torch.cat(logits_batches),
        torch.cat(label_batches),
    )


def train_qat(
    train_loader,
    val_loader,
    device: torch.device,
) -> tuple[nn.Module, dict, Path]:
    checkpoint = torch.load(
        PRODUCTION_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    model = PetBreedClassifier(
        num_classes=NUM_CLASSES,
        backbone_name="resnet18",
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train()
    prepared = prepare_qat_fx(
        model,
        get_default_qat_qconfig_mapping("x86"),
        (torch.zeros(2, 3, 224, 224),),
    ).to(device)
    optimizer = Adam(prepared.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(QAT_EPOCHS):
        prepared.train()
        loss_sum = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(prepared(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * images.shape[0]
            seen += images.shape[0]
        print(
            f"QAT epoch={epoch + 1}/{QAT_EPOCHS} "
            f"loss={loss_sum / seen:.4f}"
        )

    converted = convert_fx(prepared.cpu().eval())
    example = torch.zeros(2, 3, 224, 224)
    traced = torch.jit.trace(converted, example, check_trace=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "resnet18_qat_int8.pt"
    torch.jit.save(traced, str(output_path))
    reloaded = torch.jit.load(str(output_path), map_location="cpu").eval()
    metrics = evaluate_torch_model(reloaded, val_loader, torch.device("cpu"))
    print(f"QAT int8 validation: {metrics}")
    return reloaded, metrics, output_path


def train_distilled_student(
    train_loader,
    val_loader,
    device: torch.device,
) -> tuple[nn.Module, dict, Path]:
    if not TEACHER_CHECKPOINT.exists():
        raise FileNotFoundError(TEACHER_CHECKPOINT)
    teacher_checkpoint = torch.load(
        TEACHER_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    teacher = PetBreedClassifier(
        num_classes=NUM_CLASSES,
        backbone_name="resnet50",
        pretrained=False,
        freeze_backbone=False,
    )
    teacher.load_state_dict(teacher_checkpoint["model_state_dict"])
    teacher = teacher.to(device).eval()

    if not INITIAL_STUDENT_CHECKPOINT.exists():
        raise FileNotFoundError(INITIAL_STUDENT_CHECKPOINT)
    student = PetBreedClassifier(
        num_classes=NUM_CLASSES,
        backbone_name="mobilenet_v3_small",
        pretrained=False,
        freeze_backbone=False,
    )
    initial_student = torch.load(
        INITIAL_STUDENT_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    student.load_state_dict(initial_student["model_state_dict"])
    torch.manual_seed(20261011)
    torch.cuda.manual_seed_all(20261011)
    student = student.to(device)
    optimizer = Adam(student.parameters(), lr=5e-5)

    for epoch in range(DISTILL_EPOCHS):
        student.train()
        total_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.inference_mode():
                teacher_logits = teacher(images)
            optimizer.zero_grad(set_to_none=True)
            student_logits = student(images)
            loss = distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                temperature=DISTILLATION_TEMPERATURE,
                distillation_weight=DISTILLATION_WEIGHT,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.shape[0]
            seen += images.shape[0]
        print(
            f"Distillation epoch={epoch + 1}/{DISTILL_EPOCHS} "
            f"loss={total_loss / seen:.4f}"
        )

    metrics = evaluate_torch_model(student, val_loader, device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "mobilenet_v3_small_distilled.pt"
    torch.save(
        {
            "model_state_dict": student.cpu().state_dict(),
            "backbone_name": "mobilenet_v3_small",
            "teacher": "resnet50",
            "initial_student_checkpoint": str(
                INITIAL_STUDENT_CHECKPOINT.relative_to(PROJECT_ROOT)
            ),
            "temperature": DISTILLATION_TEMPERATURE,
            "distillation_weight": DISTILLATION_WEIGHT,
            "epochs": DISTILL_EPOCHS,
            "validation_metrics": metrics,
        },
        output_path,
    )
    return student.to(device).eval(), metrics, output_path


def create_tensorrt_runner():
    if not TENSORRT_ENGINE.exists():
        raise FileNotFoundError(TENSORRT_ENGINE)
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(TENSORRT_ENGINE.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize {TENSORRT_ENGINE}.")
    if engine.get_tensor_dtype(INPUT_NAME) != trt.DataType.HALF:
        raise RuntimeError("TensorRT engine input is not FP16.")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Could not create TensorRT execution context.")
    stream = torch.cuda.Stream()

    def run(image: torch.Tensor) -> torch.Tensor:
        image = image.to(device="cuda", dtype=torch.float16).contiguous()
        output = torch.empty(
            (1, NUM_CLASSES),
            device="cuda",
            dtype=torch.float16,
        )
        with torch.cuda.stream(stream):
            if not context.set_tensor_address(INPUT_NAME, image.data_ptr()):
                raise RuntimeError("Failed to set TensorRT input address.")
            if not context.set_tensor_address(OUTPUT_NAME, output.data_ptr()):
                raise RuntimeError("Failed to set TensorRT output address.")
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT inference execution failed.")
        stream.synchronize()
        return output.float()

    return run


def evaluate_tensorrt(run, loader) -> dict:
    logits = []
    labels = []
    for images, batch_labels in loader:
        for index, image in enumerate(images):
            logits.append(run(image.unsqueeze(0)).cpu())
            labels.append(batch_labels[index].reshape(1))
    return classification_metrics(torch.cat(logits), torch.cat(labels))


def p95_latency(
    call,
    *,
    cuda: bool,
    warmup: int = LATENCY_WARMUP,
    iterations: int = LATENCY_ITERATIONS,
) -> float:
    for _ in range(warmup):
        call()
    if cuda:
        torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        if cuda:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return float(np.percentile(samples, 95))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The optimization matrix requires CUDA.")
    torch.set_num_threads(1)
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    print(f"Optimization hardware: {gpu_name}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(20261008)
    torch.cuda.manual_seed_all(20261008)
    train_loader, val_loader = create_dataloaders(batch_size=BATCH_SIZE)
    baseline = load_resnet18(device)
    baseline_metrics = evaluate_torch_model(baseline, val_loader, device)
    print(f"ResNet-18 production validation: {baseline_metrics}")
    supervised_student = load_supervised_student(device)
    supervised_student_metrics = evaluate_torch_model(
        supervised_student,
        val_loader,
        device,
    )
    print(f"Supervised MobileNetV3-Small validation: {supervised_student_metrics}")

    pruning_sweep = []
    pruning_candidates = []
    for amount in PRUNING_AMOUNTS:
        candidate = copy.deepcopy(baseline)
        filter_sparsity = apply_structured_filter_pruning(candidate, amount=amount)
        metrics = evaluate_torch_model(candidate, val_loader, device)
        pruning_sweep.append(
            {
                "requested_amount": amount,
                "structured_filter_sparsity": filter_sparsity,
                **metrics,
            }
        )
        if metrics["top1"] >= baseline_metrics["top1"] - PRUNING_ACCURACY_DROP:
            pruning_candidates.append(
                (filter_sparsity, amount, candidate, metrics)
            )
        print(
            f"Pruning sweep amount={amount:.0%}, "
            f"filter_sparsity={filter_sparsity:.2%}, "
            f"top1={metrics['top1']:.4f}"
        )
    if not pruning_candidates:
        raise RuntimeError("No pruning candidate meets the top-1 quality gate.")
    zero_filter_ratio, pruning_amount, pruned, pruned_metrics = max(
        pruning_candidates,
        key=lambda result: result[0],
    )
    pruned_path = OUTPUT_DIR / "resnet18_structured_pruned.pt"
    torch.save(
        {
            "model_state_dict": pruned.cpu().state_dict(),
            "backbone_name": "resnet18",
            "structured_filter_sparsity": zero_filter_ratio,
            "pruning_amount": pruning_amount,
        },
        pruned_path,
    )
    pruned = pruned.to(device).eval()
    print(
        f"Selected {pruning_amount:.0%} structured pruning "
        f"({zero_filter_ratio:.2%} zeroed filters); "
        f"validation: {pruned_metrics}"
    )

    int8_session, calibration_samples = quantize_onnx()
    ptq_metrics = evaluate_onnx(int8_session, val_loader)
    print(f"INT8 PTQ validation: {ptq_metrics}")

    qat_model, qat_metrics, qat_path = train_qat(
        train_loader,
        val_loader,
        device,
    )
    student, student_metrics, student_path = train_distilled_student(
        train_loader,
        val_loader,
        device,
    )
    print(f"Distilled MobileNetV3-Small validation: {student_metrics}")

    tensorrt_run = create_tensorrt_runner()
    tensorrt_metrics = evaluate_tensorrt(tensorrt_run, val_loader)
    print(f"TensorRT FP16 validation: {tensorrt_metrics}")

    sample = next(iter(val_loader))[0][:1]
    sample_gpu = sample.to(device)
    sample_numpy = sample.numpy().astype(np.float32, copy=False)
    int8_input = int8_session.get_inputs()[0].name
    int8_output = int8_session.get_outputs()[0].name
    cpu_session_options = ort.SessionOptions()
    cpu_session_options.intra_op_num_threads = 1

    rows = [
        {
            "technique": "Baseline",
            "model": "ResNet-18",
            **baseline_metrics,
            "p95_latency_ms": p95_latency(
                lambda: baseline(sample_gpu),
                cuda=True,
            ),
            "size_mb": PRODUCTION_CHECKPOINT.stat().st_size / (1024**2),
            "hardware": gpu_name,
            "model_path": str(PRODUCTION_CHECKPOINT.relative_to(PROJECT_ROOT)),
        },
        {
            "technique": f"Structured filter pruning ({pruning_amount:.0%})",
            "model": "ResNet-18",
            **pruned_metrics,
            "p95_latency_ms": p95_latency(
                lambda: pruned(sample_gpu),
                cuda=True,
            ),
            "size_mb": pruned_path.stat().st_size / (1024**2),
            "hardware": gpu_name,
            "model_path": str(pruned_path.relative_to(PROJECT_ROOT)),
            "structured_filter_sparsity": zero_filter_ratio,
            "pruning_amount": pruning_amount,
        },
        {
            "technique": "INT8 PTQ (static, stratified calibration)",
            "model": "ResNet-18 ONNX",
            **ptq_metrics,
            "p95_latency_ms": p95_latency(
                lambda: int8_session.run(
                    [int8_output],
                    {int8_input: sample_numpy},
                ),
                cuda=False,
            ),
            "size_mb": INT8_ONNX.stat().st_size / (1024**2),
            "hardware": f"CPU ({platform.processor() or platform.machine()})",
            "model_path": str(INT8_ONNX.relative_to(PROJECT_ROOT)),
            "calibration_samples": calibration_samples,
        },
        {
            "technique": "INT8 QAT (2 epochs)",
            "model": "ResNet-18 FX",
            **qat_metrics,
            "p95_latency_ms": p95_latency(
                lambda: qat_model(sample),
                cuda=False,
            ),
            "size_mb": qat_path.stat().st_size / (1024**2),
            "hardware": f"CPU ({platform.processor() or platform.machine()})",
            "model_path": str(qat_path.relative_to(PROJECT_ROOT)),
        },
        {
            "technique": "Supervised student baseline (pre-distillation)",
            "model": "MobileNetV3-Small",
            **supervised_student_metrics,
            "p95_latency_ms": p95_latency(
                lambda: supervised_student(sample_gpu),
                cuda=True,
            ),
            "size_mb": INITIAL_STUDENT_CHECKPOINT.stat().st_size / (1024**2),
            "hardware": gpu_name,
            "model_path": str(
                INITIAL_STUDENT_CHECKPOINT.relative_to(PROJECT_ROOT)
            ),
        },
        {
            "technique": "ResNet-50 to MobileNetV3-Small distillation",
            "model": "MobileNetV3-Small",
            **student_metrics,
            "p95_latency_ms": p95_latency(
                lambda: student(sample_gpu),
                cuda=True,
            ),
            "size_mb": student_path.stat().st_size / (1024**2),
            "hardware": gpu_name,
            "model_path": str(student_path.relative_to(PROJECT_ROOT)),
            "teacher_checkpoint": str(
                TEACHER_CHECKPOINT.relative_to(PROJECT_ROOT)
            ),
            "initial_student_checkpoint": str(
                INITIAL_STUDENT_CHECKPOINT.relative_to(PROJECT_ROOT)
            ),
            "distillation_epochs": DISTILL_EPOCHS,
            "distillation_temperature": DISTILLATION_TEMPERATURE,
            "distillation_weight": DISTILLATION_WEIGHT,
        },
        {
            "technique": "TensorRT FP16",
            "model": "ResNet-18 TensorRT",
            **tensorrt_metrics,
            "p95_latency_ms": p95_latency(
                lambda: tensorrt_run(sample),
                cuda=True,
            ),
            "size_mb": TENSORRT_ENGINE.stat().st_size / (1024**2),
            "hardware": gpu_name,
            "model_path": str(TENSORRT_ENGINE.relative_to(PROJECT_ROOT)),
        },
    ]

    report = {
        "validation_examples": baseline_metrics["examples"],
        "calibration_examples_per_class": CALIBRATION_SAMPLES_PER_CLASS,
        "latency_warmup": LATENCY_WARMUP,
        "latency_iterations": LATENCY_ITERATIONS,
        "pruning_accuracy_drop_limit": PRUNING_ACCURACY_DROP,
        "pruning_sweep": pruning_sweep,
        "latency_note": (
            "Single-image inference p95 after warmup. CUDA entries synchronize "
            "the device; CPU entries use ONNX Runtime or converted FX INT8."
        ),
        "production_checkpoint_unchanged": str(
            PRODUCTION_CHECKPOINT.relative_to(PROJECT_ROOT)
        ),
        "results": rows,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nMeasured optimization matrix")
    for row in rows:
        print(
            f"{row['technique']}: top1={row['top1']:.4f}, "
            f"p95={row['p95_latency_ms']:.3f} ms, "
            f"size={row['size_mb']:.2f} MB, hardware={row['hardware']}"
        )
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
