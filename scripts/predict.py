from pathlib import Path

import torch
from PIL import Image

from bet_breed_mlops.confidence import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    load_class_names,
    load_temperature,
    predict_with_confidence,
)
from bet_breed_mlops.data import val_transform
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
# Load image
# --------------------------------------------------

def load_image(
    image_path: Path,
) -> torch.Tensor:

    image = Image.open(
        image_path
    ).convert("RGB")

    image = val_transform(image)

    image = image.unsqueeze(0)

    return image


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict() -> None:

    print(f"Device: {DEVICE}")

    # -----------------------------
    # Load model
    # -----------------------------

    model = load_model()

    # -----------------------------
    # Load image
    # -----------------------------

    image = load_image(
        IMAGE_PATH
    )

    image = image.to(
        DEVICE,
        non_blocking=True,
    )

    # -----------------------------
    # Model inference
    # -----------------------------

    with torch.no_grad():

        logits = model(image)

    # Remove batch dimension
    logits = logits.squeeze(0).cpu()

    # -----------------------------
    # Calibration
    # -----------------------------

    temperature = load_temperature()

    class_names = load_class_names()

    # -----------------------------
    # Confidence prediction
    # -----------------------------

    result = predict_with_confidence(
        logits=logits,
        temperature=temperature,
        threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        class_names=class_names,
    )

    # -----------------------------
    # Print result
    # -----------------------------

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