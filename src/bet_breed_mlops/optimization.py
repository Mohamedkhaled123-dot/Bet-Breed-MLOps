from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import prune


def apply_structured_filter_pruning(
    model: nn.Module,
    amount: float = 0.30,
) -> float:
    if not 0.0 < amount < 1.0:
        raise ValueError("Pruning amount must be between zero and one.")

    modules = dict(model.named_modules())
    total_filters = 0
    zero_filters = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue

        prune.ln_structured(
            module,
            name="weight",
            amount=amount,
            n=2,
            dim=0,
        )
        filter_mask = module.weight_mask.reshape(module.out_channels, -1).sum(1) == 0
        total_filters += module.out_channels
        zero_filters += int(filter_mask.sum().item())
        prune.remove(module, "weight")

        parent, _, leaf = name.rpartition(".")
        if leaf.startswith("conv"):
            batch_norm_leaf = f"bn{leaf.removeprefix('conv')}"
        elif leaf == "0" and parent.endswith(".downsample"):
            batch_norm_leaf = "1"
        else:
            continue

        batch_norm_name = f"{parent}.{batch_norm_leaf}" if parent else batch_norm_leaf
        batch_norm = modules.get(batch_norm_name)
        if isinstance(batch_norm, nn.BatchNorm2d):
            with torch.no_grad():
                batch_norm.weight[filter_mask] = 0
                batch_norm.bias[filter_mask] = 0
                batch_norm.running_mean[filter_mask] = 0
                batch_norm.running_var[filter_mask] = 1

    if total_filters == 0:
        raise ValueError("Model contains no convolution filters to prune.")
    return zero_filters / total_filters


def distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    temperature: float = 4.0,
    distillation_weight: float = 0.7,
) -> Tensor:
    if temperature <= 0:
        raise ValueError("Distillation temperature must be positive.")
    if not 0.0 <= distillation_weight <= 1.0:
        raise ValueError("Distillation weight must be between zero and one.")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and teacher logits must have matching shapes.")
    if student_logits.shape[0] != labels.shape[0]:
        raise ValueError("Logits and labels must have matching batch sizes.")

    teacher_probabilities = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probabilities = F.log_softmax(
        student_logits / temperature,
        dim=1,
    )
    soft_loss = F.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction="batchmean",
    ) * temperature**2
    hard_loss = F.cross_entropy(student_logits, labels)
    return (
        distillation_weight * soft_loss
        + (1.0 - distillation_weight) * hard_loss
    )


def select_stratified_indices(
    labels: Iterable[int],
    samples_per_class: int = 4,
    seed: int = 2026,
) -> list[int]:
    if samples_per_class <= 0:
        raise ValueError("Samples per class must be positive.")

    indices_by_class: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        indices_by_class.setdefault(int(label), []).append(index)

    generator = torch.Generator().manual_seed(seed)
    selected = []
    for label in sorted(indices_by_class):
        indices = indices_by_class[label]
        count = min(samples_per_class, len(indices))
        order = torch.randperm(len(indices), generator=generator)[:count].tolist()
        selected.extend(indices[position] for position in order)
    if not selected:
        raise ValueError("Calibration set selection found no labels.")
    return selected
