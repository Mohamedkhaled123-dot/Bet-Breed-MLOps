from pathlib import Path

from PIL import Image

from bet_breed_mlops.onnx_inference import ONNXInference


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


def test_onnx_uses_cpu_provider():
    inference = ONNXInference()

    providers = inference.session.get_providers()

    assert "CPUExecutionProvider" in providers


def test_onnx_cpu_predicts():
    inference = ONNXInference()

    image = Image.open(IMAGE_PATH)

    result = inference.predict(image)

    assert result["breed"]
    assert 0 <= result["confidence"] <= 1
    assert 0 <= result["class_id"] < 37