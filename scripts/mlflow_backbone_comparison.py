import argparse
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow import MlflowClient
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Adam

from bet_breed_mlops.calibration import calculate_ece, fit_temperature
from bet_breed_mlops.confidence import DEFAULT_CONFIDENCE_THRESHOLD
from bet_breed_mlops.data import create_dataloaders
from bet_breed_mlops.model import PetBreedClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "experiments" / "checkpoints"
PRODUCTION_CHECKPOINT = PROJECT_ROOT / "artifacts" / "checkpoints" / "resnet18.pt"
REPORT_DIR = PROJECT_ROOT / "reports"
TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "bet-breed-real-backbone-comparison"
REGISTERED_MODEL_NAME = "PetBreedClassifier"
NUM_CLASSES = 37
BATCH_SIZE = 32
EPOCHS = 2
MAXIMUM_QUALITY_DROP = 0.01

EXPERIMENTS = (
    ("resnet18", 1e-3),
    ("mobilenet_v3_small", 1e-3),
    ("resnet50", 1e-3),
    ("resnet18", 3e-4),
    ("mobilenet_v3_small", 3e-4),
)


def validation_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, float]:
    if logits.ndim != 2 or logits.shape[0] != labels.shape[0]:
        raise ValueError("Expected [N, classes] logits and N labels.")
    if logits.shape[0] == 0:
        raise ValueError("Validation metrics require at least one example.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    predictions = logits.argmax(dim=1)
    accuracy = float(predictions.eq(labels).float().mean().item())
    macro_f1 = float(
        f1_score(
            labels.numpy(),
            predictions.numpy(),
            labels=list(range(logits.shape[1])),
            average="macro",
            zero_division=0,
        )
    )
    temperature = fit_temperature(logits, labels)
    calibrated_logits = logits / temperature
    probabilities = calibrated_logits.softmax(dim=1)
    confidences, calibrated_predictions = probabilities.max(dim=1)
    accepted = confidences >= threshold
    accepted_count = int(accepted.sum().item())
    selective_accuracy = (
        float(
            calibrated_predictions[accepted]
            .eq(labels[accepted])
            .float()
            .mean()
            .item()
        )
        if accepted_count
        else 0.0
    )

    return {
        "validation_examples": float(logits.shape[0]),
        "val_top1": accuracy,
        "val_macro_f1": macro_f1,
        "val_ece": calculate_ece(logits, labels),
        "val_calibrated_ece": calculate_ece(calibrated_logits, labels),
        "temperature": temperature,
        "coverage": float(accepted.float().mean().item()),
        "selective_accuracy": selective_accuracy,
    }


def passes_quality_gate(
    candidate_top1: float,
    production_top1: float,
    maximum_drop: float = MAXIMUM_QUALITY_DROP,
) -> bool:
    if not 0.0 <= candidate_top1 <= 1.0:
        raise ValueError("Candidate top-1 must be between 0 and 1.")
    if not 0.0 <= production_top1 <= 1.0:
        raise ValueError("Production top-1 must be between 0 and 1.")
    if not 0.0 <= maximum_drop <= 1.0:
        raise ValueError("Maximum drop must be between 0 and 1.")
    return candidate_top1 >= production_top1 - maximum_drop


def collect_logits(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits = []
    all_labels = []
    with torch.inference_mode():
        for images, labels in loader:
            all_logits.append(model(images.to(device, non_blocking=True)).cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def train_experiment(
    backbone_name: str,
    learning_rate: float,
    seed: int,
    train_loader,
    val_loader,
    device: torch.device,
) -> tuple[dict[str, float], Path, int]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = PetBreedClassifier(
        num_classes=NUM_CLASSES,
        backbone_name=backbone_name,
        pretrained=True,
        freeze_backbone=True,
    ).to(device)
    optimizer = Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )
    criterion = nn.CrossEntropyLoss()
    epoch_losses = []

    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0.0
        example_count = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * images.shape[0]
            example_count += images.shape[0]
        epoch_losses.append(loss_sum / example_count)
        print(
            f"{backbone_name} lr={learning_rate:g} "
            f"epoch={epoch + 1}/{EPOCHS} loss={epoch_losses[-1]:.4f}"
        )

    logits, labels = collect_logits(model, val_loader, device)
    metrics = validation_metrics(logits, labels)
    metrics["train_loss"] = epoch_losses[-1]

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{backbone_name}_lr_{learning_rate:g}.pt"
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "backbone_name": backbone_name,
            "num_classes": NUM_CLASSES,
            "learning_rate": learning_rate,
            "epochs": EPOCHS,
            "train_loss": epoch_losses[-1],
            "val_metrics": metrics,
        },
        checkpoint_path,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return metrics, checkpoint_path, parameter_count


def log_training_run(
    backbone_name: str,
    learning_rate: float,
    seed: int,
    metrics: dict[str, float],
    checkpoint_path: Path,
    parameter_count: int,
    duration_seconds: float,
) -> dict:
    run_name = f"{backbone_name}-lr-{learning_rate:g}"
    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_params(
            {
                "backbone": backbone_name,
                "pretrained": True,
                "freeze_backbone": True,
                "batch_size": BATCH_SIZE,
                "learning_rate": learning_rate,
                "seed": seed,
                "epochs": EPOCHS,
                "optimizer": "Adam",
                "image_size": 224,
                "num_classes": NUM_CLASSES,
                "device": "cuda",
            }
        )
        mlflow.log_metrics(
            {
                **metrics,
                "parameters": float(parameter_count),
                "training_seconds": duration_seconds,
                "checkpoint_size_mb": checkpoint_path.stat().st_size / (1024**2),
            }
        )
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoint")
        return {
            "run_id": active_run.info.run_id,
            "run_name": run_name,
            "backbone": backbone_name,
            "learning_rate": learning_rate,
            "seed": seed,
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            **metrics,
            "parameters": parameter_count,
            "training_seconds": duration_seconds,
            "checkpoint_size_mb": checkpoint_path.stat().st_size / (1024**2),
        }


def evaluate_production(
    val_loader,
    device: torch.device,
) -> tuple[dict[str, float], nn.Module, str]:
    checkpoint = torch.load(
        PRODUCTION_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("backbone_name") != "resnet18":
        raise ValueError("Production checkpoint is not the ResNet-18 candidate.")

    model = PetBreedClassifier(
        num_classes=NUM_CLASSES,
        backbone_name="resnet18",
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    logits, labels = collect_logits(model, val_loader, device)
    metrics = validation_metrics(logits, labels)

    with mlflow.start_run(run_name="resnet18-production-baseline") as active_run:
        mlflow.log_params(
            {
                "backbone": "resnet18",
                "checkpoint": str(PRODUCTION_CHECKPOINT.relative_to(PROJECT_ROOT)),
                "epochs": checkpoint["epoch"],
                "device": "cuda",
                "role": "production-baseline",
                "num_classes": NUM_CLASSES,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(PRODUCTION_CHECKPOINT), artifact_path="checkpoint")
        return metrics, model, active_run.info.run_id


def promote_production_model(model: nn.Module, run_id: str) -> int:
    signature = ModelSignature(
        inputs=Schema(
            [
                TensorSpec(
                    np.dtype("float32"),
                    (-1, 3, 224, 224),
                    "image",
                )
            ]
        ),
        outputs=Schema(
            [
                TensorSpec(
                    np.dtype("float32"),
                    (-1, NUM_CLASSES),
                    "logits",
                )
            ]
        ),
    )
    with mlflow.start_run(run_id=run_id):
        model_info = mlflow.pytorch.log_model(
            model.cpu(),
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            serialization_format="pt2",
            input_example=torch.zeros(2, 3, 224, 224),
            signature=signature,
            await_registration_for=60,
        )
    version = int(model_info.registered_model_version)
    client = MlflowClient()
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        "Production",
        str(version),
    )
    print(f"Registered production model version {version} from run {run_id}.")
    return version


def write_comparison_report(rows: list[dict], production_top1: float) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    passing_rows = [
        row
        for row in rows
        if passes_quality_gate(row["val_top1"], production_top1)
    ]
    best = max(passing_rows, key=lambda row: row["val_top1"])
    report = {
        "experiment": EXPERIMENT_NAME,
        "validation_examples": int(rows[0]["validation_examples"]),
        "production_baseline_top1": production_top1,
        "quality_gate": (
            f"candidate top-1 >= production top-1 - {MAXIMUM_QUALITY_DROP}"
        ),
        "selected_candidate": best["run_name"],
        "selected_candidate_top1": best["val_top1"],
        "registered_production_backbone": "resnet18",
        "runs": rows,
    }
    (REPORT_DIR / "mlflow_comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    names = [row["run_name"] for row in rows]
    top1 = [row["val_top1"] for row in rows]
    macro_f1 = [row["val_macro_f1"] for row in rows]
    locations = np.arange(len(rows))
    width = 0.38
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(locations - width / 2, top1, width, label="Top-1")
    axis.bar(locations + width / 2, macro_f1, width, label="Macro F1")
    axis.axhline(
        production_top1 - MAXIMUM_QUALITY_DROP,
        color="red",
        linestyle="--",
        label="Production quality-gate floor",
    )
    axis.set(
        title="Measured MLflow validation comparison",
        ylabel="Validation score",
        ylim=(0, 1),
        xticks=locations,
        xticklabels=names,
    )
    axis.tick_params(axis="x", labelrotation=25)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "mlflow_comparison.png", dpi=160)
    plt.close(figure)


def load_production_model() -> nn.Module:
    checkpoint = torch.load(
        PRODUCTION_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    model = PetBreedClassifier(
        num_classes=NUM_CLASSES,
        backbone_name="resnet18",
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--register-existing",
        action="store_true",
        help="Register the production baseline from a completed comparison report.",
    )
    args = parser.parse_args()
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    if args.register_existing:
        report_path = REPORT_DIR / "mlflow_comparison.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        baseline = next(
            row
            for row in report["runs"]
            if row["run_name"] == "resnet18-production-baseline"
        )
        version = promote_production_model(
            load_production_model(),
            baseline["run_id"],
        )
        print(f"Registered production baseline as model version {version}.")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("MLflow model experiments require CUDA.")
    device = torch.device("cuda")
    print(f"Training and evaluating on {torch.cuda.get_device_name(device)}")

    train_loader, val_loader = create_dataloaders(batch_size=BATCH_SIZE)
    print(
        f"Train examples: {len(train_loader.dataset)}; "
        f"validation examples: {len(val_loader.dataset)}"
    )

    production_metrics, production_model, production_run_id = evaluate_production(
        val_loader, device
    )
    rows = []
    for index, (backbone_name, learning_rate) in enumerate(EXPERIMENTS):
        start = time.perf_counter()
        metrics, checkpoint_path, parameter_count = train_experiment(
            backbone_name,
            learning_rate,
            20261008 + index,
            train_loader,
            val_loader,
            device,
        )
        row = log_training_run(
            backbone_name,
            learning_rate,
            20261008 + index,
            metrics,
            checkpoint_path,
            parameter_count,
            time.perf_counter() - start,
        )
        row["quality_gate_passed"] = passes_quality_gate(
            row["val_top1"], production_metrics["val_top1"]
        )
        rows.append(row)
        print(
            f"Measured {row['run_name']}: "
            f"top1={row['val_top1']:.4f}, "
            f"macro_f1={row['val_macro_f1']:.4f}, "
            f"ECE={row['val_ece']:.4f}, "
            f"coverage={row['coverage']:.4f}, "
            f"selective_accuracy={row['selective_accuracy']:.4f}"
        )

    rows.append(
        {
            "run_id": production_run_id,
            "run_name": "resnet18-production-baseline",
            "backbone": "resnet18",
            "learning_rate": None,
            "checkpoint": str(PRODUCTION_CHECKPOINT.relative_to(PROJECT_ROOT)),
            **production_metrics,
            "quality_gate_passed": True,
        }
    )
    write_comparison_report(rows, production_metrics["val_top1"])
    promote_production_model(production_model, production_run_id)
    print(
        f"Completed {len(rows)} real MLflow runs; "
        f"production top-1={production_metrics['val_top1']:.4f}."
    )


if __name__ == "__main__":
    main()
