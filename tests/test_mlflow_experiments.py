import pytest
import torch

from bet_breed_mlops.model import PetBreedClassifier
from scripts.mlflow_backbone_comparison import (
    passes_quality_gate,
    validation_metrics,
)


def test_resnet50_classifier_outputs_37_logits() -> None:
    model = PetBreedClassifier(
        num_classes=37,
        backbone_name="resnet50",
        pretrained=False,
    )
    output = model(torch.zeros(1, 3, 224, 224))
    assert output.shape == (1, 37)


def test_validation_metrics_report_measured_selective_results() -> None:
    logits = torch.tensor(
        [
            [4.0, 0.0, -1.0],
            [0.0, 3.0, -1.0],
            [0.2, 0.0, 0.1],
        ]
    )
    labels = torch.tensor([0, 1, 2])

    metrics = validation_metrics(logits, labels, threshold=0.8)

    assert metrics["validation_examples"] == 3
    assert metrics["val_top1"] == pytest.approx(2 / 3)
    assert 0.0 <= metrics["val_macro_f1"] <= 1.0
    assert 0.0 <= metrics["val_ece"] <= 1.0
    assert 0.0 <= metrics["coverage"] <= 1.0
    assert 0.0 <= metrics["selective_accuracy"] <= 1.0
    assert metrics["temperature"] > 0


def test_quality_gate_allows_at_most_one_percentage_point_drop() -> None:
    assert passes_quality_gate(0.89, 0.90)
    assert not passes_quality_gate(0.889, 0.90)


@pytest.mark.parametrize(
    ("candidate", "production"),
    [(-0.1, 0.8), (1.1, 0.8), (0.8, 1.1)],
)
def test_quality_gate_rejects_invalid_accuracies(
    candidate: float,
    production: float,
) -> None:
    with pytest.raises(ValueError):
        passes_quality_gate(candidate, production)
