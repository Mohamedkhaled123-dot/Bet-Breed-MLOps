
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from bet_breed_mlops.confidence import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    load_class_names,
    load_class_species,
    load_temperature,
)
from bet_breed_mlops.data import eval_transform

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ONNX_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "resnet18.onnx"
)

TEMPERATURE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "calibration"
    / "temperature.pt"
)


class ONNXInference:
    def __init__(
        self,
        onnx_path: Path = ONNX_PATH,
        temperature_path: Path = TEMPERATURE_PATH,
        threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")

        self.onnx_path = onnx_path
        self.temperature_path = temperature_path
        self.threshold = threshold

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.temperature = load_temperature()
        if self.temperature <= 0:
            raise ValueError("Temperature must be greater than zero.")

        self.class_names = load_class_names()
        self.class_species = load_class_species()

    def predict(self, image: Image.Image) -> dict:
        image = image.convert("RGB")
        input_tensor = eval_transform(image).unsqueeze(0)

        input_numpy = input_tensor.numpy().astype(np.float32)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_numpy},
        )

        logits = outputs[0][0]
        calibrated_logits = logits / self.temperature
        probabilities = self._softmax(calibrated_logits)

        # Highest-probability classes first.
        top_indices = np.argsort(probabilities)[::-1][:3]
        predicted_class = int(top_indices[0])
        confidence = float(probabilities[predicted_class])

        is_confident = confidence >= self.threshold

        top_3 = [
            {
                "breed": self.class_names[int(class_index)],
                "species": self.class_species[int(class_index)],
                "confidence": float(probabilities[class_index]),
            }
            for class_index in top_indices
        ]

        return {
            "breed": (
                self.class_names[predicted_class]
                if is_confident
                else None
            ),
            "species": (
                self.class_species[predicted_class]
                if is_confident
                else None
            ),
            "confidence": confidence,
            "top_3": top_3,
            "decision": "confident" if is_confident else "uncertain",
            "threshold": self.threshold,
            "status": "Confident" if is_confident else "Not sure",
            "class_id": predicted_class,
        }

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values)
