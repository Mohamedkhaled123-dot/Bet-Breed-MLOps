import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bet_breed_mlops.confidence import MANIFEST_PATH
from bet_breed_mlops.data import PetBreedDataset, eval_transform
from bet_breed_mlops.model import PetBreedClassifier
from mlflow_backbone_comparison import passes_quality_gate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "checkpoints" / "resnet18.pt"
BASELINE_REPORT_PATH = PROJECT_ROOT / "reports" / "mlflow_comparison.json"
QUALITY_REPORT_PATH = PROJECT_ROOT / "reports" / "quality_gate.json"
MAXIMUM_TOP1_DROP = 0.01
BATCH_SIZE = 32


def main() -> None:
    baseline_report = json.loads(
        BASELINE_REPORT_PATH.read_text(encoding="utf-8")
    )
    production_top1 = float(baseline_report["production_baseline_top1"])
    dataset = PetBreedDataset(MANIFEST_PATH, "val", transform=eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )
    model = PetBreedClassifier(
        num_classes=37,
        backbone_name="resnet18",
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    correct = 0
    example_count = 0
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images).argmax(dim=1)
            correct += int(predictions.eq(labels).sum().item())
            example_count += len(labels)
    if example_count == 0:
        raise RuntimeError("Quality gate received an empty validation dataset.")
    candidate_top1 = correct / example_count
    quality_gate_floor = production_top1 - MAXIMUM_TOP1_DROP
    result = {
        "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
        "validation_examples": example_count,
        "production_baseline_top1": production_top1,
        "candidate_top1": candidate_top1,
        "maximum_top1_drop": MAXIMUM_TOP1_DROP,
        "quality_gate_floor": quality_gate_floor,
        "passed": passes_quality_gate(
            candidate_top1,
            production_top1,
            MAXIMUM_TOP1_DROP,
        ),
    }
    QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_REPORT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("Production quality gate failed.")


if __name__ == "__main__":
    main()
