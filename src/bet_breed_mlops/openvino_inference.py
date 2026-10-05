from pathlib import Path

import numpy as np
import openvino as ov
from PIL import Image

from bet_breed_mlops.confidence import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    load_class_names,
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


class OpenVINOInference:
    def __init__(
        self,
        onnx_path: Path = ONNX_PATH,
        temperature_path: Path = TEMPERATURE_PATH,
        threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.onnx_path = onnx_path
        self.temperature_path = temperature_path
        self.threshold = threshold

        self.core = ov.Core()

        model = self.core.read_model(
            str(self.onnx_path)
        )

        self.compiled_model = self.core.compile_model(
            model,
            "CPU",
        )

        self.infer_request = (
            self.compiled_model.create_infer_request()
        )

        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        self.temperature = load_temperature()
        self.class_names = load_class_names()

    def predict(self, image: Image.Image) -> dict:
        image = image.convert("RGB")

        input_tensor = (
            eval_transform(image)
            .unsqueeze(0)
            .numpy()
            .astype(np.float32)
        )

        result = self.infer_request.infer(
            {
                self.input_layer: input_tensor,
            }
        )

        logits = np.asarray(
            result[self.output_layer]
        )[0]

        calibrated_logits = (
            logits / self.temperature
        )

        probabilities = self._softmax(
            calibrated_logits
        )

        predicted_class = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[predicted_class]
        )

        breed = self.class_names[predicted_class]

        if confidence >= self.threshold:
            status = "Confident"
        else:
            status = "Not sure"

        return {
            "breed": breed,
            "confidence": confidence,
            "threshold": self.threshold,
            "status": status,
            "class_id": predicted_class,
        }

    @staticmethod
    def _softmax(
        logits: np.ndarray,
    ) -> np.ndarray:
        shifted = logits - np.max(logits)

        exp_values = np.exp(shifted)

        return exp_values / np.sum(exp_values)