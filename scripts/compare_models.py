from pathlib import Path
import time

import torch

from bet_breed_mlops.model import PetBreedClassifier


BACKBONES = [
    "resnet18",
    "mobilenet_v3_small",
    "efficientnet_b0",
]

CHECKPOINT_DIR = Path("artifacts/checkpoints")
IMAGE_SIZE = 224
WARMUP_RUNS = 20
BENCHMARK_RUNS = 100


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def get_checkpoint_size(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def benchmark_inference(
    model: torch.nn.Module,
    device: torch.device,
) -> float:
    dummy_input = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
    )

    model.eval()

    with torch.inference_mode():
        # Warm-up
        for _ in range(WARMUP_RUNS):
            _ = model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(BENCHMARK_RUNS):
            _ = model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    return (elapsed / BENCHMARK_RUNS) * 1000


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print("\nModel Comparison")
    print("================")

    for backbone in BACKBONES:
        checkpoint_path = CHECKPOINT_DIR / f"{backbone}.pt"

        model = PetBreedClassifier(
            backbone_name=backbone,
            pretrained=False,
            freeze_backbone=True,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        parameters = count_parameters(model)
        size_mb = get_checkpoint_size(checkpoint_path)
        inference_ms = benchmark_inference(model, device)

        print(f"\n{backbone}")
        print("-" * len(backbone))
        print(f"Parameters      : {parameters:,}")
        print(f"Checkpoint Size : {size_mb:.2f} MB")
        print(f"Inference Time  : {inference_ms:.2f} ms/image")


if __name__ == "__main__":
    main()