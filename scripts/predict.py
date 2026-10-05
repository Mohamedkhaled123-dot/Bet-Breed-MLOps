from pathlib import Path

import torch
from PIL import Image

from bet_breed_mlops.confidence import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    load_temperature,
    predict_image,
)
from bet_breed_mlops.model import PetBreedClassifier


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "resnet18_baseline.pt"
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


# --------------------------------------------------
# Load model
# --------------------------------------------------

def load_model() -> torch.nn.Module:

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

    return model


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict() -> None:

    print(f"Device: {DEVICE}")

    model = load_model()

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    result = predict_image(
        model=model,
        image=image,
        threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    )

    temperature = load_temperature()

    print()
    print("Prediction")
    print("==========")

    print(
        f"Image      : {IMAGE_PATH.name}"
    )

    print(
        f"Temperature: {temperature:.4f}"
    )

    print(
        f"Threshold  : "
        f"{DEFAULT_CONFIDENCE_THRESHOLD:.2f}"
    )

    print(
        f"Class      : "
        f"{result.class_index}"
    )

    print(
        f"Breed      : "
        f"{result.breed}"
    )

    print(
        f"Confidence : "
        f"{result.confidence:.4f}"
    )

    print(
        f"Status     : "
        f"{'Confident' if result.is_confident else 'Not sure'}"
    )


if __name__ == "__main__":
    predict()