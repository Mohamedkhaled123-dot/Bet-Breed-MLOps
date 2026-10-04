from pathlib import Path

import torch

from bet_breed_mlops.calibration import (
    DEVICE,
    CHECKPOINT_PATH,
    collect_validation_logits,
)
from bet_breed_mlops.data import create_dataloaders
from bet_breed_mlops.model import PetBreedClassifier


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CALIBRATION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "calibration"
    / "temperature.pt"
)


# Candidate confidence thresholds
THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]


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
# Load temperature
# --------------------------------------------------

def load_temperature() -> float:

    calibration = torch.load(
        CALIBRATION_PATH,
        map_location="cpu",
    )

    return float(
        calibration["temperature"]
    )


# --------------------------------------------------
# Threshold analysis
# --------------------------------------------------

def analyze_thresholds(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> None:

    # Apply temperature scaling
    calibrated_logits = (
        logits / temperature
    )

    probabilities = torch.softmax(
        calibrated_logits,
        dim=1,
    )

    confidences, predictions = probabilities.max(
        dim=1
    )

    correct = predictions.eq(labels)

    total_samples = len(labels)

    print()
    print("Threshold Analysis")
    print("==================")

    print()
    print(
        f"{'Threshold':<12}"
        f"{'Coverage':<12}"
        f"{'Accepted':<12}"
        f"{'Accuracy':<12}"
    )

    print("-" * 48)

    for threshold in THRESHOLDS:

        accepted = confidences >= threshold

        accepted_count = int(
            accepted.sum().item()
        )

        coverage = (
            accepted_count
            / total_samples
        )

        if accepted_count > 0:

            accuracy = (
                correct[accepted]
                .float()
                .mean()
                .item()
            )

        else:
            accuracy = 0.0

        print(
            f"{threshold:<12.2f}"
            f"{coverage:<12.4f}"
            f"{accepted_count:<12}"
            f"{accuracy:<12.4f}"
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def analyze() -> None:

    print(f"Device: {DEVICE}")

    # -----------------------------
    # Data
    # -----------------------------

    _, val_loader = create_dataloaders(
        batch_size=32
    )

    # -----------------------------
    # Model
    # -----------------------------

    model = load_model()

    # -----------------------------
    # Validation logits
    # -----------------------------

    print()
    print("Collecting validation logits...")

    logits, labels = collect_validation_logits(
        model,
        val_loader,
    )

    # -----------------------------
    # Temperature
    # -----------------------------

    temperature = load_temperature()

    print(
        f"Temperature: {temperature:.4f}"
    )

    # -----------------------------
    # Analyze
    # -----------------------------

    analyze_thresholds(
        logits=logits,
        labels=labels,
        temperature=temperature,
    )


if __name__ == "__main__":
    analyze()