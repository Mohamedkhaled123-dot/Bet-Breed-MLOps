import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import openvino as ov
from PIL import Image

from bet_breed_mlops.data import eval_transform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports" / "cpu_inference_benchmark.json"

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

    return tensor.numpy().astype(np.float32)


def benchmark_onnxruntime(
    input_numpy: np.ndarray,
) -> list[float]:

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=[
            "CPUExecutionProvider",
        ],
    )

    input_name = session.get_inputs()[0].name

    for _ in range(WARMUP_RUNS):
        session.run(
            None,
            {input_name: input_numpy},
        )

    times = []

    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()

        session.run(
            None,
            {input_name: input_numpy},
        )

        end = time.perf_counter()

        times.append(
            (end - start) * 1000
        )

    return times


def benchmark_openvino(
    input_numpy: np.ndarray,
) -> list[float]:

    core = ov.Core()

    model = core.read_model(
        str(ONNX_PATH)
    )

    compiled_model = core.compile_model(
        model,
        "CPU",
    )

    infer_request = (
        compiled_model.create_infer_request()
    )

    input_layer = compiled_model.input(0)

    for _ in range(WARMUP_RUNS):
        infer_request.infer(
            {
                input_layer: input_numpy,
            }
        )

    times = []

    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()

        infer_request.infer(
            {
                input_layer: input_numpy,
            }
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

    values = np.asarray(times)

    mean = np.mean(values)
    p50 = np.percentile(values, 50)
    p95 = np.percentile(values, 95)
    p99 = np.percentile(values, 99)

    throughput = 1000 / mean

    print()
    print(name)
    print("=" * len(name))
    print(f"Mean latency : {mean:.3f} ms")
    print(f"P50 latency  : {p50:.3f} ms")
    print(f"P95 latency  : {p95:.3f} ms")
    print(f"P99 latency  : {p99:.3f} ms")
    print(f"Throughput   : {throughput:.2f} images/sec")


def summarize(name: str, times: list[float]) -> dict:
    values = np.asarray(times, dtype=np.float64)
    return {
        "runtime": name,
        "device": "CPU",
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": BENCHMARK_RUNS,
        "mean_latency_ms": float(np.mean(values)),
        "p50_latency_ms": float(np.percentile(values, 50)),
        "p95_latency_ms": float(np.percentile(values, 95)),
        "p99_latency_ms": float(np.percentile(values, 99)),
        "throughput_images_per_second": float(1000 / np.mean(values)),
    }


def main() -> None:

    print("CPU Runtime Benchmark")
    print("=====================")
    print(f"Warmup runs     : {WARMUP_RUNS}")
    print(f"Benchmark runs  : {BENCHMARK_RUNS}")

    input_numpy = load_input()

    ort_times = benchmark_onnxruntime(
        input_numpy
    )

    openvino_times = benchmark_openvino(
        input_numpy
    )

    print_results(
        "ONNX Runtime CPU",
        ort_times,
    )

    print_results(
        "OpenVINO CPU",
        openvino_times,
    )
    report = {
        "onnx_model": str(ONNX_PATH.relative_to(PROJECT_ROOT)),
        "input_image": str(IMAGE_PATH.relative_to(PROJECT_ROOT)),
        "input_shape": [1, 3, 224, 224],
        "results": [
            summarize("ONNX Runtime", ort_times),
            summarize("OpenVINO", openvino_times),
        ],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()