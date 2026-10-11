from pathlib import Path
import json

import torch

from bet_breed_mlops.calibration import (
    DEVICE,
    collect_validation_logits,
)
from bet_breed_mlops.confidence import (
    load_temperature,
    select_confidence_threshold,
)
from bet_breed_mlops.data import create_dataloaders
from bet_breed_mlops.model import PetBreedClassifier


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "resnet18.pt"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "confidence_evaluation.json"
)

MINIMUM_SELECTIVE_ACCURACY = 0.98

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
# Evaluate thresholds
# --------------------------------------------------

def evaluate_thresholds(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> list[dict]:

    calibrated_logits = logits / temperature

    probabilities = torch.softmax(
        calibrated_logits,
        dim=1,
    )

    confidences, predictions = probabilities.max(
        dim=1
    )

    correct = predictions.eq(labels)

    total_samples = len(labels)

    results = []

    for threshold in THRESHOLDS:

        accepted = confidences >= threshold

        accepted_count = int(
            accepted.sum().item()
        )

        not_sure_count = (
            total_samples - accepted_count
        )

        coverage = (
            accepted_count
            / total_samples
        )

        if accepted_count > 0:

            accepted_accuracy = (
                correct[accepted]
                .float()
                .mean()
                .item()
            )

        else:

            accepted_accuracy = 0.0

        results.append(
            {
                "threshold": threshold,
                "total_samples": total_samples,
                "accepted": accepted_count,
                "not_sure": not_sure_count,
                "coverage": coverage,
                "accepted_accuracy": accepted_accuracy,
            }
        )

    return results


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------

def evaluate() -> None:

    print(f"Device: {DEVICE}")

    # -----------------------------
    # Validation data
    # -----------------------------

    _, val_loader = create_dataloaders(
        batch_size=32
    )

    print(
        f"Validation samples: "
        f"{len(val_loader.dataset)}"
    )

    # -----------------------------
    # Model
    # -----------------------------

    model = load_model()

    # -----------------------------
    # Collect logits
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
    # Evaluate thresholds
    # -----------------------------

    results = evaluate_thresholds(
        logits=logits,
        labels=labels,
        temperature=temperature,
    )
    selected_threshold = select_confidence_threshold(
        results,
        minimum_selective_accuracy=MINIMUM_SELECTIVE_ACCURACY,
    )

    # -----------------------------
    # Print report
    # -----------------------------

    print()
    print("Confidence / Abstention Evaluation")
    print("=" * 70)

    print(
        f"{'Threshold':<12}"
        f"{'Accepted':<12}"
        f"{'Not Sure':<12}"
        f"{'Coverage':<12}"
        f"{'Accuracy':<12}"
    )

    print("-" * 58)

    for result in results:

        print(
            f"{result['threshold']:<12.2f}"
            f"{result['accepted']:<12}"
            f"{result['not_sure']:<12}"
            f"{result['coverage']:<12.4f}"
            f"{result['accepted_accuracy']:<12.4f}"
        )

    print()
    print(
        "Selected threshold: "
        f"{selected_threshold['threshold']:.2f} "
        f"(coverage={selected_threshold['coverage']:.4f}, "
        "selective_accuracy="
        f"{selected_threshold['accepted_accuracy']:.4f}, "
        f"minimum={MINIMUM_SELECTIVE_ACCURACY:.2f})"
    )

    # -----------------------------
    # Save report
    # -----------------------------

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "dataset": "validation",
        "num_samples": len(labels),
        "temperature": temperature,
        "minimum_selective_accuracy": MINIMUM_SELECTIVE_ACCURACY,
        "selected_threshold": selected_threshold,
        "selection_rule": (
            "Maximum validation coverage among listed thresholds "
            "whose selective accuracy meets the required minimum."
        ),
        "results": results,
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print()
    print(
        f"Report saved -> {REPORT_PATH}"
    )


if __name__ == "__main__":
    evaluate()