from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch

from bet_breed_mlops.data import eval_transform
from bet_breed_mlops.model import PetBreedClassifier
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "resnet18.pt"
)

ONNX_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "resnet18.onnx"
)

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "split"
    / "val"
    / "Abyssinian"
    / "Abyssinian_1.jpg"
)

WARMUP_RUNS = 20
BENCHMARK_RUNS = 100


def load_input() -> np.ndarray:
    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    tensor = eval_transform(
        image
    ).unsqueeze(0)

    return tensor.numpy()


def benchmark_pytorch(
    input_numpy: np.ndarray,
) -> list[float]:

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = PetBreedClassifier(
        num_classes=37,
        pretrained=False,
        freeze_backbone=True,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    input_tensor = torch.from_numpy(
        input_numpy
    ).to(device)

    # Warmup
    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model(input_tensor)

    torch.cuda.synchronize()

    times = []

    with torch.inference_mode():
        for _ in range(BENCHMARK_RUNS):

            start = time.perf_counter()

            model(input_tensor)

            torch.cuda.synchronize()

            end = time.perf_counter()

            times.append(
                (end - start) * 1000
            )

    return times


def benchmark_onnx(
    input_numpy: np.ndarray,
) -> list[float]:

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(WARMUP_RUNS):
        session.run(
            None,
            {
                input_name: input_numpy
            },
        )

    times = []

    for _ in range(BENCHMARK_RUNS):

        start = time.perf_counter()

        session.run(
            None,
            {
                input_name: input_numpy
            },
        )

        end = time.perf_counter()

        times.append(
            (end - start) * 1000
        )

    return times


def print_results(
    name: str,
    times: list[float],
) -> None:

    times_array = np.array(times)

    mean_latency = np.mean(
        times_array
    )

    median_latency = np.percentile(
        times_array,
        50,
    )

    p95_latency = np.percentile(
        times_array,
        95,
    )

    throughput = (
        1000 / mean_latency
    )

    print()
    print(name)
    print("=" * len(name))
    print(
        f"Mean latency : {mean_latency:.3f} ms"
    )
    print(
        f"P50 latency  : {median_latency:.3f} ms"
    )
    print(
        f"P95 latency  : {p95_latency:.3f} ms"
    )
    print(
        f"Throughput   : {throughput:.2f} images/sec"
    )


def main() -> None:

    print("ONNX vs PyTorch GPU Benchmark")
    print("==============================")
    print(f"Warmup runs : {WARMUP_RUNS}")
    print(f"Benchmark runs: {BENCHMARK_RUNS}")

    input_numpy = load_input()

    pytorch_times = benchmark_pytorch(
        input_numpy
    )

    onnx_times = benchmark_onnx(
        input_numpy
    )

    print_results(
        "PyTorch GPU",
        pytorch_times,
    )

    print_results(
        "ONNX GPU",
        onnx_times,
    )


if __name__ == "__main__":
    main()