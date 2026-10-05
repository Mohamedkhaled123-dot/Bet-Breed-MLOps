from pathlib import Path

import numpy as np
from PIL import Image

from bet_breed_mlops.openvino_inference import (
    OpenVINOInference,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "split"
    / "val"
    / "Abyssinian"
    / "Abyssinian_1.jpg"
)


def test_openvino_initializes():
    inference = OpenVINOInference()

    assert inference.compiled_model is not None
    assert inference.input_layer is not None
    assert inference.output_layer is not None


def test_openvino_predicts():
    inference = OpenVINOInference()

    image = Image.open(IMAGE_PATH)

    result = inference.predict(image)

    assert "breed" in result
    assert "confidence" in result
    assert "class_id" in result
    assert "status" in result

    assert 0 <= result["confidence"] <= 1
    assert 0 <= result["class_id"] < 37


def test_openvino_handles_rgb_conversion():
    inference = OpenVINOInference()

    image = Image.open(IMAGE_PATH).convert("L")

    result = inference.predict(image)

    assert result["breed"]
    assert 0 <= result["confidence"] <= 1