from pathlib import Path

from bet_breed_mlops.bento_service import (
    PetBreedService,
    PredictRequest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = PROJECT_ROOT / "data" / "raw" / "split" / "test"


def get_test_image():
    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    images = sorted(
        path
        for path in TEST_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
    )

    assert images, "No test images found."

    return images[0]


def make_request():
    image_path = get_test_image()

    image_base64 = __import__("base64").b64encode(
        image_path.read_bytes()
    ).decode("utf-8")

    return PredictRequest(image=image_base64)


def test_bento_single_prediction():
    service = PetBreedService()

    request = make_request()

    result = service.predict([request])

    assert len(result) == 1

    prediction = result[0]

    assert "class_index" in prediction
    assert "breed" in prediction
    assert "confidence" in prediction
    assert "is_confident" in prediction

    assert isinstance(prediction["class_index"], int)
    assert 0.0 <= prediction["confidence"] <= 1.0


def test_bento_batch_prediction():
    service = PetBreedService()

    request = make_request()

    batch = [request] * 4

    results = service.predict(batch)

    assert len(results) == 4

    for prediction in results:
        assert "class_index" in prediction
        assert "breed" in prediction
        assert "confidence" in prediction
        assert "is_confident" in prediction

        assert isinstance(prediction["class_index"], int)
        assert 0.0 <= prediction["confidence"] <= 1.0