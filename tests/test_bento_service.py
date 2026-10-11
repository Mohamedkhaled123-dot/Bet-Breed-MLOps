import base64
from io import BytesIO

import bentoml
from PIL import Image
import pytest

from bet_breed_mlops.bento_model import register_bento_model


@pytest.fixture(scope="module")
def bento_runtime():
    model = register_bento_model()
    from bet_breed_mlops.bento_service import (
        MODEL_VERSION,
        onnx_runner,
        predict_batch,
        svc,
        PredictRequest,
    )

    onnx_runner.init_local()
    yield model, onnx_runner, predict_batch, PredictRequest, MODEL_VERSION, svc
    onnx_runner.destroy()


def make_request(PredictRequest):
    image = Image.new("RGB", (224, 224), color=(128, 96, 64))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return PredictRequest(image=encoded)


def test_bento_model_is_registered_and_runner_is_batchable(bento_runtime):
    model, runner, _, _, _, service = bento_runtime

    assert bentoml.models.get(str(model.tag)).tag == model.tag
    assert model.info.signatures["run"].batchable is True
    assert model.info.signatures["run"].batch_dim == (0, 0)
    assert runner in service.runners


def test_bento_runner_scores_a_batch(bento_runtime):
    _, runner, predict_batch, request_type, model_version, _ = bento_runtime
    requests = [make_request(request_type) for _ in range(4)]

    results = predict_batch(requests, runner)

    assert len(results) == 4
    for result in results:
        assert result["model_version"] == model_version
        assert result["decision"] in {"confident", "uncertain"}
        assert result["class_id"] in range(37)
        assert len(result["top_3"]) == 3
        assert 0.0 <= result["confidence"] <= 1.0
        assert all(
            0.0 <= prediction["confidence"] <= 1.0
            for prediction in result["top_3"]
        )
