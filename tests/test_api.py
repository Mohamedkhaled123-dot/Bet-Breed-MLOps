from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from bet_breed_mlops import api


class FakeInference:
    class_names = {index: f"breed-{index}" for index in range(37)}

    def predict(self, image: Image.Image) -> dict:
        assert image.mode == "RGB"
        return {
            "breed": "Bengal",
            "species": "cat",
            "confidence": 0.91,
            "threshold": 0.80,
            "status": "Confident",
            "class_id": 5,
            "top_3": [
                {
                    "breed": "Bengal",
                    "species": "cat",
                    "confidence": 0.91,
                },
                {
                    "breed": "Abyssinian",
                    "species": "cat",
                    "confidence": 0.06,
                },
                {
                    "breed": "Birman",
                    "species": "cat",
                    "confidence": 0.02,
                },
            ],
            "decision": "confident",
        }


@pytest.fixture
def client():
    test_app = api.create_app(FakeInference)
    with TestClient(test_app) as test_client:
        yield test_client


def encode_image(image: Image.Image, format: str = "PNG") -> BytesIO:
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return buffer


def test_health_contract(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_version": "resnet18-onnx-cpu-v1",
        "num_classes": 37,
    }


def test_metrics_endpoint_exposes_prometheus_inference_series(
    client: TestClient,
):
    buffer = encode_image(Image.new("RGB", (32, 32), color="white"))
    prediction = client.post(
        "/predict",
        files={"file": ("metrics.png", buffer, "image/png")},
    )
    response = client.get("/metrics")

    assert prediction.status_code == 200
    assert response.status_code == 200
    assert "pet_breed_prediction_requests_total" in response.text
    assert "pet_breed_prediction_p95_seconds" in response.text


def test_predict_valid_image_returns_handbook_schema(client: TestClient):
    buffer = encode_image(Image.new("RGB", (224, 224), color="white"))

    response = client.post(
        "/predict",
        files={"file": ("test.png", buffer, "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "filename": "test.png",
        "breed": "Bengal",
        "species": "cat",
        "confidence": 0.91,
        "top_3": [
            {"breed": "Bengal", "species": "cat", "confidence": 0.91},
            {"breed": "Abyssinian", "species": "cat", "confidence": 0.06},
            {"breed": "Birman", "species": "cat", "confidence": 0.02},
        ],
        "decision": "confident",
        "model_version": "resnet18-onnx-cpu-v1",
        "threshold": 0.80,
        "status": "Confident",
        "class_id": 5,
    }


def test_predict_converts_four_channel_png_to_rgb(client: TestClient):
    buffer = encode_image(Image.new("RGBA", (224, 224), color=(20, 30, 40, 128)))

    response = client.post(
        "/predict",
        files={"file": ("alpha.png", buffer, "image/png")},
    )

    assert response.status_code == 200


def test_predict_rejects_invalid_image_with_422(client: TestClient):
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 422


def test_predict_rejects_oversized_upload_with_413(client: TestClient):
    response = client.post(
        "/predict",
        files={
            "file": (
                "large.png",
                b"x" * (api.MAX_UPLOAD_BYTES + 1),
                "image/png",
            )
        },
    )

    assert response.status_code == 413
