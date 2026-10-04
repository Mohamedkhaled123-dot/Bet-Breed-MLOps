import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from bet_breed_mlops.data import create_dataloaders
from bet_breed_mlops.model import PetBreedClassifier


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "manifest.json"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
)

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SUPPORTED_BACKBONES = {
    "resnet18",
    "mobilenet_v3_small",
    "efficientnet_b0",
}


# --------------------------------------------------
# Load label map
# --------------------------------------------------

def load_class_names() -> dict[int, str]:

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        manifest = json.load(file)

    label_map = manifest["label_map"]

    return {
        int(class_index): breed
        for breed, class_index in label_map.items()
    }


# --------------------------------------------------
# Print most confused pairs
# --------------------------------------------------

def print_confused_pairs(
    confusion,
    class_names: dict[int, str],
    top_k: int = 10,
) -> None:

    pairs = []

    num_classes = confusion.shape[0]

    for true_class in range(num_classes):

        for predicted_class in range(num_classes):

            if true_class == predicted_class:
                continue

            errors = confusion[
                true_class,
                predicted_class,
            ]

            if errors > 0:

                pairs.append(
                    (
                        int(errors),
                        true_class,
                        predicted_class,
                    )
                )

    pairs.sort(reverse=True)

    print()
    print("Most Confused Class Pairs")
    print("-------------------------")

    for (
        errors,
        true_class,
        predicted_class,
    ) in pairs[:top_k]:

        true_name = class_names[
            true_class
        ]

        predicted_name = class_names[
            predicted_class
        ]

        print(
            f"{true_name} -> "
            f"{predicted_name}: "
            f"{errors} errors"
        )


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(
    backbone_name: str,
) -> None:

    if backbone_name not in SUPPORTED_BACKBONES:

        raise ValueError(
            f"Unsupported backbone: "
            f"{backbone_name}"
        )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Backbone: {backbone_name}"
    )

    # -----------------------------
    # Checkpoint
    # -----------------------------

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"{backbone_name}.pt"
    )

    # -----------------------------
    # Data
    # -----------------------------

    _, val_loader = (
        create_dataloaders(
            batch_size=32
        )
    )

    # -----------------------------
    # Model
    # -----------------------------

    model = PetBreedClassifier(
        num_classes=37,
        backbone_name=backbone_name,
        pretrained=False,
        freeze_backbone=True,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)

    model.eval()

    # -----------------------------
    # Predictions
    # -----------------------------

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    # -----------------------------
    # Metrics
    # -----------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
    )

    print()
    print("Validation Results")
    print("------------------")

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Macro-F1 : {macro_f1:.4f}"
    )

    # -----------------------------
    # Confusion Matrix
    # -----------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(37)),
    )

    confusion_matrix_path = (
        REPORTS_DIR
        / f"confusion_matrix_{backbone_name}.png"
    )

    plt.figure(
        figsize=(14, 12)
    )

    plt.imshow(cm)

    plt.title(
        f"{backbone_name} - "
        "Validation Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "True Class"
    )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        confusion_matrix_path,
        dpi=200,
    )

    plt.close()

    print()

    print(
        f"Confusion matrix saved → "
        f"{confusion_matrix_path}"
    )

    # -----------------------------
    # Confusion analysis
    # -----------------------------

    class_names = load_class_names()

    print_confused_pairs(
        cm,
        class_names,
        top_k=10,
    )


# --------------------------------------------------
# Command Line Interface
# --------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a pet breed "
            "classifier on validation data."
        )
    )

    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=sorted(
            SUPPORTED_BACKBONES
        ),
        help="Backbone architecture.",
    )

    return parser.parse_args()


# --------------------------------------------------
# Entry Point
# --------------------------------------------------

if __name__ == "__main__":

    args = parse_args()

    evaluate(
        backbone_name=args.backbone
    )