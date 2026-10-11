import argparse
from pathlib import Path

import torch
from torch import nn, optim

from bet_breed_mlops.data import create_dataloaders
from bet_breed_mlops.model import PetBreedClassifier


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SUPPORTED_BACKBONES = {
    "resnet18",
    "resnet50",
    "mobilenet_v3_small",
    "efficientnet_b0",
}


# --------------------------------------------------
# Validation
# --------------------------------------------------

def validate(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
) -> tuple[float, float]:

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = (
        total_loss / total
    )

    accuracy = correct / total

    return average_loss, accuracy


# --------------------------------------------------
# Training
# --------------------------------------------------

def train(
    backbone_name: str,
) -> None:

    if backbone_name not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone: {backbone_name}. "
            f"Choose from: {sorted(SUPPORTED_BACKBONES)}"
        )

    print(f"Device: {DEVICE}")
    print(f"Backbone: {backbone_name}")

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

    train_loader, val_loader = (
        create_dataloaders(
            batch_size=BATCH_SIZE
        )
    )

    print(
        f"Train samples: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_loader.dataset)}"
    )

    # -----------------------------
    # Model
    # -----------------------------

    model = PetBreedClassifier(
        num_classes=37,
        backbone_name=backbone_name,
        pretrained=True,
        freeze_backbone=True,
    )

    model = model.to(DEVICE)

    # -----------------------------
    # Loss + Optimizer
    # -----------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
    )

    # -----------------------------
    # Training
    # -----------------------------

    best_val_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Clear gradients
            optimizer.zero_grad()

            # Forward
            outputs = model(images)

            # Loss
            loss = criterion(
                outputs,
                labels,
            )

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            # Statistics
            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = (
            running_loss / total
        )

        train_accuracy = (
            correct / total
        )

        # -----------------------------
        # Validation
        # -----------------------------

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
        )

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # -----------------------------
        # Save best model
        # -----------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = (
                val_accuracy
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "val_accuracy":
                        val_accuracy,

                    "epoch":
                        epoch + 1,

                    "num_classes":
                        37,

                    "backbone_name":
                        backbone_name,
                },
                checkpoint_path,
            )

            print(
                f"Best model saved → "
                f"{checkpoint_path}"
            )

    print()
    print("Training finished.")

    print(
        f"Backbone: {backbone_name}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )


# --------------------------------------------------
# Command Line Interface
# --------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Train Oxford-IIIT Pet breed "
            "classifier."
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

    train(
        backbone_name=args.backbone
    )