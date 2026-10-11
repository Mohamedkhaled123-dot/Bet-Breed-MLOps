from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

CALIBRATION_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "calibration_metrics.json"
)

RELIABILITY_DIAGRAM_PATH = (
    PROJECT_ROOT
    / "reports"
    / "calibration.png"
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


def reliability_bins(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> tuple[list[float], list[float], list[int]]:
    probabilities = torch.softmax(logits, dim=1)
    confidences, predictions = probabilities.max(dim=1)
    accuracies = predictions.eq(labels)
    boundaries = torch.linspace(0.0, 1.0, n_bins + 1)

    mean_confidences = []
    mean_accuracies = []
    bin_counts = []

    for index in range(n_bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        if index == n_bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)

        count = int(in_bin.sum().item())
        if count:
            mean_confidences.append(
                float(confidences[in_bin].mean().item())
            )
            mean_accuracies.append(
                float(accuracies[in_bin].float().mean().item())
            )
            bin_counts.append(count)

    return mean_confidences, mean_accuracies, bin_counts


def save_reliability_diagram(
    logits: torch.Tensor,
    labels: torch.Tensor,
    calibrated_logits: torch.Tensor,
    output_path: Path = RELIABILITY_DIAGRAM_PATH,
) -> Path:
    before_confidence, before_accuracy, _ = reliability_bins(
        logits,
        labels,
    )
    after_confidence, after_accuracy, _ = reliability_bins(
        calibrated_logits,
        labels,
    )

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for axis, confidences, accuracies, title in (
        (axes[0], before_confidence, before_accuracy, "Before temperature scaling"),
        (axes[1], after_confidence, after_accuracy, "After temperature scaling"),
    ):
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        axis.plot(confidences, accuracies, marker="o", label="ResNet-18")
        axis.set(
            title=f"{title} (ECE={calculate_ece(logits if axis is axes[0] else calibrated_logits, labels):.4f})",
            xlabel="Mean confidence",
            ylabel="Empirical accuracy",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right")

    figure.suptitle("Validation-set reliability diagram (15 equal-width bins)")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


# --------------------------------------------------
# Fit Temperature
# --------------------------------------------------

def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:

    log_temperature = nn.Parameter(
        torch.zeros(1)
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.01,
        max_iter=50,
    )

    def closure():

        optimizer.zero_grad()

        temperature = log_temperature.clamp(
            min=-4.605170186,
            max=2.302585093,
        ).exp()
        scaled_logits = logits / temperature

        loss = criterion(
            scaled_logits,
            labels,
        )

        loss.backward()

        return loss

    optimizer.step(closure)

    return float(
        log_temperature.detach()
        .clamp(min=-4.605170186, max=2.302585093)
        .exp()
        .item()
    )


# --------------------------------------------------
# Calibration
# --------------------------------------------------

def calibrate() -> None:

    print(f"Device: {DEVICE}")
    if DEVICE.type != "cuda":
        raise RuntimeError(
            "Calibration requires CUDA for this project run."
        )

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

    diagram_path = save_reliability_diagram(
        logits=logits,
        labels=labels,
        calibrated_logits=calibrated_logits,
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

    CALIBRATION_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with CALIBRATION_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
                "dataset": "validation",
                "num_samples": int(labels.numel()),
                "device": torch.cuda.get_device_name(DEVICE),
                "temperature": temperature,
                "ece_before": ece_before,
                "ece_after": ece_after,
                "reliability_diagram": str(
                    diagram_path.relative_to(PROJECT_ROOT)
                ),
                "reliability_bins": 15,
            },
            file,
            indent=2,
        )

    print()
    print(
        "Calibration saved -> "
        f"{CALIBRATION_PATH}"
    )
    print(f"Reliability diagram saved -> {diagram_path}")
    print(f"Calibration report saved -> {CALIBRATION_REPORT_PATH}")


if __name__ == "__main__":
    calibrate()