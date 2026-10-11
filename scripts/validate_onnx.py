import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader, Subset

from bet_breed_mlops.confidence import MANIFEST_PATH
from bet_breed_mlops.data import PetBreedDataset, eval_transform
from bet_breed_mlops.model import PetBreedClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "checkpoints" / "resnet18.pt"
ONNX_PATH = PROJECT_ROOT / "artifacts" / "models" / "resnet18.onnx"
REPORT_PATH = PROJECT_ROOT / "reports" / "onnx_parity.json"
NUM_IMAGES = 200
BATCH_SIZE = 32
LOGIT_TOLERANCE = 1e-4


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch/ONNX parity validation requires CUDA.")
    cuda_device = torch.device("cuda")
    torch.set_num_threads(1)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    dataset = PetBreedDataset(MANIFEST_PATH, "test", transform=eval_transform)
    if len(dataset) < NUM_IMAGES:
        raise RuntimeError(
            f"ONNX parity requires {NUM_IMAGES} test images; found {len(dataset)}."
        )
    loader = DataLoader(
        Subset(dataset, range(NUM_IMAGES)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )
    model_cpu = PetBreedClassifier(
        num_classes=37,
        backbone_name="resnet18",
        pretrained=False,
        freeze_backbone=False,
    )
    model_cpu.load_state_dict(checkpoint["model_state_dict"])
    model_cpu.eval()
    model_cuda = PetBreedClassifier(
        num_classes=37,
        backbone_name="resnet18",
        pretrained=False,
        freeze_backbone=False,
    )
    model_cuda.load_state_dict(checkpoint["model_state_dict"])
    model_cuda.to(cuda_device).eval()

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )
    cuda_session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=[
            ("CUDAExecutionProvider", {"use_tf32": "0"}),
            "CPUExecutionProvider",
        ],
    )
    if "CUDAExecutionProvider" not in cuda_session.get_providers():
        raise RuntimeError(
            "ONNX Runtime did not activate CUDAExecutionProvider."
        )
    input_name = session.get_inputs()[0].name
    cuda_input_name = cuda_session.get_inputs()[0].name
    cpu_max_absolute_difference = 0.0
    cpu_sum_absolute_difference = 0.0
    cuda_max_absolute_difference = 0.0
    cuda_sum_absolute_difference = 0.0
    cuda_vs_cpu_max_absolute_difference = 0.0
    cuda_vs_cpu_sum_absolute_difference = 0.0
    cpu_matching_predictions = 0
    cuda_matching_predictions = 0
    value_count = 0
    image_count = 0
    with torch.inference_mode():
        for images, _ in loader:
            pytorch_cpu_logits = model_cpu(images).numpy()
            pytorch_cuda_logits = model_cuda(
                images.to(cuda_device, non_blocking=True)
            )
            torch.cuda.synchronize(cuda_device)
            pytorch_cuda_values = pytorch_cuda_logits.cpu().numpy()
            onnx_logits = session.run(
                None,
                {input_name: images.numpy().astype(np.float32, copy=False)},
            )[0]
            onnx_cuda_logits = cuda_session.run(
                None,
                {
                    cuda_input_name: images.numpy().astype(
                        np.float32,
                        copy=False,
                    )
                },
            )[0]
            cpu_absolute_difference = np.abs(pytorch_cpu_logits - onnx_logits)
            cuda_absolute_difference = np.abs(
                pytorch_cuda_values - onnx_cuda_logits
            )
            cuda_vs_cpu_difference = np.abs(
                pytorch_cuda_values - onnx_logits
            )
            cpu_max_absolute_difference = max(
                cpu_max_absolute_difference,
                float(cpu_absolute_difference.max()),
            )
            cpu_sum_absolute_difference += float(cpu_absolute_difference.sum())
            cuda_max_absolute_difference = max(
                cuda_max_absolute_difference,
                float(cuda_absolute_difference.max()),
            )
            cuda_sum_absolute_difference += float(cuda_absolute_difference.sum())
            cuda_vs_cpu_max_absolute_difference = max(
                cuda_vs_cpu_max_absolute_difference,
                float(cuda_vs_cpu_difference.max()),
            )
            cuda_vs_cpu_sum_absolute_difference += float(
                cuda_vs_cpu_difference.sum()
            )
            cpu_matching_predictions += int(
                np.equal(
                    pytorch_cpu_logits.argmax(axis=1),
                    onnx_logits.argmax(axis=1),
                ).sum()
            )
            cuda_matching_predictions += int(
                np.equal(
                    pytorch_cuda_values.argmax(axis=1),
                    onnx_cuda_logits.argmax(axis=1),
                ).sum()
            )
            value_count += cpu_absolute_difference.size
            image_count += len(images)

    report = {
        "images": image_count,
        "batch_size": BATCH_SIZE,
        "onnx_provider": "CPUExecutionProvider",
        "onnx_cuda_providers": cuda_session.get_providers(),
        "cpu_pytorch_vs_onnx": {
            "pytorch_device": "CPU",
            "max_absolute_logit_difference": cpu_max_absolute_difference,
            "mean_absolute_logit_difference": cpu_sum_absolute_difference
            / value_count,
            "matching_top1_predictions": cpu_matching_predictions,
            "top1_agreement": cpu_matching_predictions / image_count,
            "passed": cpu_max_absolute_difference <= LOGIT_TOLERANCE,
        },
        "cuda_pytorch_vs_onnx": {
            "pytorch_device": torch.cuda.get_device_name(cuda_device),
            "onnx_provider": "CUDAExecutionProvider",
            "max_absolute_logit_difference": cuda_max_absolute_difference,
            "mean_absolute_logit_difference": cuda_sum_absolute_difference
            / value_count,
            "matching_top1_predictions": cuda_matching_predictions,
            "top1_agreement": cuda_matching_predictions / image_count,
            "passed": cuda_max_absolute_difference <= LOGIT_TOLERANCE,
        },
        "cuda_pytorch_vs_onnx_cpu_diagnostic": {
            "max_absolute_logit_difference": cuda_vs_cpu_max_absolute_difference,
            "mean_absolute_logit_difference": cuda_vs_cpu_sum_absolute_difference
            / value_count,
        },
        "logit_tolerance": LOGIT_TOLERANCE,
        "passed": (
            cpu_max_absolute_difference <= LOGIT_TOLERANCE
            and cuda_max_absolute_difference <= LOGIT_TOLERANCE
        ),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise AssertionError(
            "PyTorch and ONNX logits differ by more than "
            f"{LOGIT_TOLERANCE}: CPU max diff={cpu_max_absolute_difference}, "
            f"CUDA max diff={cuda_max_absolute_difference}"
        )


if __name__ == "__main__":
    main()
