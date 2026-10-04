from pathlib import Path

import torch
import torch.nn as nn

from bet_breed_mlops.data import create_dataloaders
from bet_breed_mlops.model import PetBreedClassifier


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "resnet18.pt"
)

CALIBRATION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "calibration"
    / "temperature.pt"
)


# --------------------------------------------------
# Collect validation logits
# --------------------------------------------------

def collect_validation_logits(
    model: nn.Module,
    val_loader,
) -> tuple[torch.Tensor, torch.Tensor]:

    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            logits = model(images)

            all_logits.append(
                logits.cpu()
            )

            all_labels.append(
                labels
            )

    logits = torch.cat(
        all_logits
    )

    labels = torch.cat(
        all_labels
    )

    return logits, labels


# --------------------------------------------------
# Expected Calibration Error
# --------------------------------------------------

def calculate_ece(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    confidences, predictions = probabilities.max(
        dim=1
    )

    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = torch.tensor(0.0)

    for index in range(n_bins):

        lower = bin_boundaries[index]
        upper = bin_boundaries[index + 1]

        if index == n_bins - 1:

            in_bin = (
                (confidences >= lower)
                & (confidences <= upper)
            )

        else:

            in_bin = (
                (confidences >= lower)
                & (confidences < upper)
            )

        if in_bin.any():

            bin_accuracy = accuracies[
                in_bin
            ].float().mean()

            bin_confidence = confidences[
                in_bin
            ].mean()

            bin_size = in_bin.float().mean()

            ece += (
                torch.abs(
                    bin_confidence
                    - bin_accuracy
                )
                * bin_size
            )

    return ece.item()


# --------------------------------------------------
# Fit Temperature
# --------------------------------------------------

def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:

    temperature = nn.Parameter(
        torch.ones(1)
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.LBFGS(
        [temperature],
        lr=0.01,
        max_iter=50,
    )

    def closure():

        optimizer.zero_grad()

        scaled_logits = (
            logits / temperature.clamp(
                min=0.01,
                max=10.0,
            )
        )

        loss = criterion(
            scaled_logits,
            labels,
        )

        loss.backward()

        return loss

    optimizer.step(closure)

    return temperature.item()


# --------------------------------------------------
# Calibration
# --------------------------------------------------

def calibrate() -> None:

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

    # -----------------------------
    # Collect logits
    # -----------------------------

    print()
    print("Collecting validation logits...")

    logits, labels = collect_validation_logits(
        model,
        val_loader,
    )

    print(
        f"Logits shape: {logits.shape}"
    )

    print(
        f"Labels shape: {labels.shape}"
    )

    # -----------------------------
    # ECE before calibration
    # -----------------------------

    ece_before = calculate_ece(
        logits,
        labels,
    )

    print()
    print(
        f"ECE before calibration: "
        f"{ece_before:.4f}"
    )

    # -----------------------------
    # Fit temperature
    # -----------------------------

    print()
    print("Fitting temperature...")

    temperature = fit_temperature(
        logits,
        labels,
    )

    print(
        f"Temperature: {temperature:.4f}"
    )

    # -----------------------------
    # ECE after calibration
    # -----------------------------

    calibrated_logits = (
        logits / temperature
    )

    ece_after = calculate_ece(
        calibrated_logits,
        labels,
    )

    print()
    print(
        f"ECE after calibration: "
        f"{ece_after:.4f}"
    )

    # -----------------------------
    # Save temperature
    # -----------------------------

    CALIBRATION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "temperature": temperature,
            "ece_before": ece_before,
            "ece_after": ece_after,
        },
        CALIBRATION_PATH,
    )

    print()
    print(
        f"Calibration saved → "
        f"{CALIBRATION_PATH}"
    )


if __name__ == "__main__":
    calibrate()