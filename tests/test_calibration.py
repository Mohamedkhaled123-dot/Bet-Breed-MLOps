from pathlib import Path

import torch

from bet_breed_mlops.calibration import (
    calculate_ece,
    reliability_bins,
    save_reliability_diagram,
)


def test_reliability_bins_cover_nonempty_bins():
    logits = torch.tensor(
        [
            [4.0, 0.0],
            [0.0, 4.0],
            [0.5, 0.4],
        ]
    )
    labels = torch.tensor([0, 1, 0])

    confidences, accuracies, counts = reliability_bins(
        logits,
        labels,
        n_bins=5,
    )

    assert sum(counts) == len(labels)
    assert len(confidences) == len(accuracies) == len(counts)
    assert all(0.0 <= value <= 1.0 for value in confidences)
    assert all(0.0 <= value <= 1.0 for value in accuracies)


def test_reliability_diagram_is_saved(tmp_path: Path):
    logits = torch.tensor(
        [
            [4.0, 0.0],
            [0.0, 4.0],
            [0.5, 0.4],
        ]
    )
    labels = torch.tensor([0, 1, 0])
    calibrated_logits = logits / 2.0
    output_path = tmp_path / "calibration.png"

    result = save_reliability_diagram(
        logits,
        labels,
        calibrated_logits,
        output_path,
    )

    assert result == output_path
    assert output_path.is_file()
    assert output_path.read_bytes().startswith(b"\x89PNG")
    assert calculate_ece(logits, labels) >= 0
