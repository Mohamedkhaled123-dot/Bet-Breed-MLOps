from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from bet_breed_mlops.api import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_image():
    image = Image.new(
        "RGB",
        (224, 224),
        color="white",
    )

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.jpg",
                buffer,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "filename" in data
    assert "breed" in data
    assert "confidence" in data
    assert "threshold" in data
    assert "status" in data
    assert "class_id" in data

    assert 0.0 <= data["confidence"] <= 1.0
    assert data["threshold"] == 0.85
    assert data["status"] in {
        "Confident",
        "Not sure",
    }
    assert isinstance(data["class_id"], int)


def test_predict_invalid_file():
    response = client.post(
        "/predict",
        files={
            "file": (
                "test.txt",
                b"this is not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400