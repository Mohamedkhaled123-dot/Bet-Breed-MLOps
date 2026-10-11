import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import torch
from mlflow import MlflowClient
from mlflow_backbone_comparison import (
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
    TRACKING_URI,
    collect_logits,
    load_production_model,
    passes_quality_gate,
    promote_production_model,
    validation_metrics,
)
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from bet_breed_mlops.confidence import MANIFEST_PATH
from bet_breed_mlops.corruptions import CORRUPTIONS
from bet_breed_mlops.data import (
    PetBreedDataset,
    create_dataloaders,
    train_transform,
)
from bet_breed_mlops.model import PetBreedClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CHECKPOINT = PROJECT_ROOT / "artifacts" / "checkpoints" / "resnet18.pt"
RETRAINING_DIR = PROJECT_ROOT / "artifacts" / "retraining"
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 1e-5
NUM_CLASSES = 37


def safe_batch_id(batch_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", batch_id).strip("-")
    if not normalized:
        raise ValueError("batch_id must contain a safe identifier character.")
    return normalized[:80]


class CorruptedTrainingDataset(Dataset):
    def __init__(self, dataset: PetBreedDataset) -> None:
        self.records = dataset.records
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = PROJECT_ROOT / record["path"]
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
            corruption_names = sorted(CORRUPTIONS)
            corruption = corruption_names[(index + self.epoch) % len(corruption_names)]
            severity = ((index + self.epoch) % 3) + 1
            image = CORRUPTIONS[corruption](image, severity)
            image = train_transform(image)
        return image, int(record["class_index"])


def artifact_paths(batch_id: str) -> tuple[Path, Path]:
    normalized = safe_batch_id(batch_id)
    return (
        RETRAINING_DIR / f"{normalized}_candidate.pt",
        RETRAINING_DIR / f"{normalized}_evaluation.json",
    )


def train_candidate(batch_id: str) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Drift retraining requires CUDA.")
    torch.manual_seed(20261008)
    torch.cuda.manual_seed_all(20261008)
    torch.set_num_threads(1)
    device = torch.device("cuda")
    checkpoint = torch.load(
        PRODUCTION_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    model = PetBreedClassifier(
        num_classes=NUM_CLASSES,
        backbone_name="resnet18",
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    training_data = CorruptedTrainingDataset(
        PetBreedDataset(MANIFEST_PATH, "train", transform=None)
    )
    loader = DataLoader(
        training_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    epoch_losses = []
    started = time.perf_counter()
    for epoch in range(EPOCHS):
        training_data.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        example_count = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * images.shape[0]
            example_count += images.shape[0]
        if example_count == 0:
            raise RuntimeError("Drift retraining received no training examples.")
        epoch_losses.append(loss_sum / example_count)
        torch.cuda.synchronize(device)
        print(
            f"corruption-mix epoch={epoch + 1}/{EPOCHS} "
            f"loss={epoch_losses[-1]:.5f}"
        )

    candidate_path, evaluation_path = artifact_paths(batch_id)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "backbone_name": "resnet18",
            "num_classes": NUM_CLASSES,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "epoch_losses": epoch_losses,
            "training_distribution": "balanced mixed corruption types and severities 1-3",
            "source_checkpoint": str(
                PRODUCTION_CHECKPOINT.relative_to(PROJECT_ROOT)
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        candidate_path,
    )
    return {
        "batch_id": batch_id,
        "candidate_checkpoint": str(candidate_path),
        "evaluation_path": str(evaluation_path),
        "training_examples": len(training_data),
        "epochs": EPOCHS,
        "epoch_losses": epoch_losses,
        "training_seconds": time.perf_counter() - started,
        "device": torch.cuda.get_device_name(device),
    }


def evaluate_candidate(batch_id: str) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Drift candidate evaluation requires CUDA.")
    torch.set_num_threads(1)
    device = torch.device("cuda")
    candidate_path, evaluation_path = artifact_paths(batch_id)
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    checkpoint = torch.load(candidate_path, map_location="cpu", weights_only=False)
    candidate = PetBreedClassifier(
        num_classes=NUM_CLASSES,
        backbone_name="resnet18",
        pretrained=False,
        freeze_backbone=False,
    )
    candidate.load_state_dict(checkpoint["model_state_dict"])
    candidate.to(device)
    production = load_production_model().to(device)
    _, validation_loader = create_dataloaders(batch_size=BATCH_SIZE)
    candidate_logits, labels = collect_logits(
        candidate,
        validation_loader,
        device,
    )
    production_logits, production_labels = collect_logits(
        production,
        validation_loader,
        device,
    )
    if not torch.equal(labels, production_labels):
        raise RuntimeError("Candidate and production validation labels differ.")
    candidate_metrics = validation_metrics(candidate_logits, labels)
    production_metrics = validation_metrics(production_logits, production_labels)
    passed = passes_quality_gate(
        candidate_metrics["val_top1"],
        production_metrics["val_top1"],
    )
    result = {
        "batch_id": batch_id,
        "candidate_checkpoint": str(candidate_path),
        "candidate_metrics": candidate_metrics,
        "production_top1": production_metrics["val_top1"],
        "quality_gate_floor": production_metrics["val_top1"] - 0.01,
        "quality_gate_passed": passed,
        "decision": "promote" if passed else "reject",
    }
    evaluation_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def promote_or_reject(batch_id: str) -> dict:
    candidate_path, evaluation_path = artifact_paths(batch_id)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if not evaluation["quality_gate_passed"]:
        return {
            "batch_id": batch_id,
            "decision": "rejected",
            "quality_gate_floor": evaluation["quality_gate_floor"],
            "candidate_top1": evaluation["candidate_metrics"]["val_top1"],
            "production_checkpoint_preserved": str(
                PRODUCTION_CHECKPOINT.relative_to(PROJECT_ROOT)
            ),
        }
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(f"{EXPERIMENT_NAME}-drift-retraining")
    checkpoint = torch.load(candidate_path, map_location="cpu", weights_only=False)
    model = PetBreedClassifier(
        num_classes=NUM_CLASSES,
        backbone_name="resnet18",
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    with mlflow.start_run(run_name=f"drift-retraining-{safe_batch_id(batch_id)}") as run:
        mlflow.log_params(
            {
                "backbone": "resnet18",
                "training_distribution": checkpoint["training_distribution"],
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "device": "cuda",
                "source_checkpoint": str(
                    PRODUCTION_CHECKPOINT.relative_to(PROJECT_ROOT)
                ),
            }
        )
        mlflow.log_metrics(evaluation["candidate_metrics"])
        mlflow.log_metrics(
            {
                "production_top1": evaluation["production_top1"],
                "quality_gate_floor": evaluation["quality_gate_floor"],
            }
        )
        mlflow.log_artifact(str(candidate_path), artifact_path="checkpoint")
        run_id = run.info.run_id
    version = promote_production_model(model, run_id)
    MlflowClient().set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        "Production",
        str(version),
    )
    return {
        "batch_id": batch_id,
        "decision": "promoted",
        "registered_model": REGISTERED_MODEL_NAME,
        "model_version": version,
        "run_id": run_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "evaluate", "promote"))
    parser.add_argument("--batch-id", required=True)
    args = parser.parse_args()
    actions = {
        "train": train_candidate,
        "evaluate": evaluate_candidate,
        "promote": promote_or_reject,
    }
    result = actions[args.action](args.batch_id)
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
