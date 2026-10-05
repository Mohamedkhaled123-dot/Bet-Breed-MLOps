from pathlib import Path

import torch

from bet_breed_mlops.model import PetBreedClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "resnet18.pt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "resnet18.onnx"
)


def export_onnx() -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = PetBreedClassifier(
        num_classes=37,
        pretrained=False,
        freeze_backbone=True,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    dummy_input = torch.randn(
        1,
        3,
        224,
        224,
        device=DEVICE,
    )

    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_PATH,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=18,
    )

    print("ONNX export successful")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    export_onnx()