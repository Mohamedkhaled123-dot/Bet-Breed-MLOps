import pytest
import torch
from torch import nn

from bet_breed_mlops.optimization import (
    apply_structured_filter_pruning,
    distillation_loss,
    select_stratified_indices,
)


def test_structured_pruning_zeros_whole_convolution_filters() -> None:
    model = nn.Sequential(nn.Conv2d(3, 10, kernel_size=3, bias=False))
    ratio = apply_structured_filter_pruning(model, amount=0.3)

    filter_norms = model[0].weight.flatten(1).norm(p=2, dim=1)
    assert ratio == pytest.approx(0.3)
    assert int(filter_norms.eq(0).sum()) == 3


def test_structured_pruning_rejects_invalid_amount() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        apply_structured_filter_pruning(nn.Linear(2, 2), amount=1.1)


def test_distillation_loss_is_finite_and_differentiable() -> None:
    student_logits = torch.randn(4, 3, requires_grad=True)
    teacher_logits = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 1])

    loss = distillation_loss(student_logits, teacher_logits, labels)
    loss.backward()

    assert torch.isfinite(loss)
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()


def test_stratified_calibration_selection_covers_each_class() -> None:
    labels = [0, 0, 0, 1, 1, 2, 2, 2, 2]

    selected = select_stratified_indices(labels, samples_per_class=2, seed=12)

    assert len(selected) == 6
    assert {labels[index] for index in selected} == {0, 1, 2}
