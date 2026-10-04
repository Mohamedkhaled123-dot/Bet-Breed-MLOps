from dataclasses import dataclass
from pathlib import Path
import json

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "manifest.json"
)

CALIBRATION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "calibration"
    / "temperature.pt"
)


# Operational default.
# This is configurable and is NOT a "best" threshold.
DEFAULT_CONFIDENCE_THRESHOLD = 0.80


@dataclass(frozen=True)
class PredictionResult:
    class_index: int
    confidence: float
    breed: str | None
    is_confident: bool


def load_temperature() -> float:
    calibration = torch.load(
        CALIBRATION_PATH,
        map_location="cpu",
    )

    return float(calibration["temperature"])


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


def apply_temperature(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:

    if temperature <= 0:
        raise ValueError(
            "Temperature must be greater than zero."
        )

    return logits / temperature


def predict_with_confidence(
    logits: torch.Tensor,
    temperature: float,
    threshold: float,
    class_names: dict[int, str],
) -> PredictionResult:

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    calibrated_logits = apply_temperature(
        logits,
        temperature,
    )

    probabilities = torch.softmax(
        calibrated_logits,
        dim=0,
    )

    confidence, predicted_class = probabilities.max(
        dim=0
    )

    class_index = int(
        predicted_class.item()
    )

    confidence_value = float(
        confidence.item()
    )

    is_confident = (
        confidence_value >= threshold
    )

    breed = (
        class_names[class_index]
        if is_confident
        else None
    )

    return PredictionResult(
        class_index=class_index,
        confidence=confidence_value,
        breed=breed,
        is_confident=is_confident,
    )

def predict_image(
    model,
    image,
    threshold: float = 0.85,
):
    """
    Run the complete inference pipeline:
    RGB -> eval_transform -> model -> temperature scaling
    -> confidence threshold -> breed / not sure.
    """
    from bet_breed_mlops.data import eval_transform

    temperature = load_temperature()
    class_names = load_class_names()

    image = image.convert("RGB")

    input_tensor = eval_transform(image).unsqueeze(0)

    with torch.inference_mode():
        logits = model(input_tensor)
        logits = logits.squeeze(0)


    result = predict_with_confidence(
        logits=logits,
        temperature=temperature,
        threshold=threshold,
        class_names=class_names,
    )

    return result