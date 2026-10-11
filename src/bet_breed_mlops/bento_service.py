import base64
from io import BytesIO

import bentoml
import numpy as np
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from bet_breed_mlops.bento_model import BENTO_MODEL_NAME
from bet_breed_mlops.confidence import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    load_class_names,
    load_class_species,
    load_temperature,
)
from bet_breed_mlops.data import eval_transform


MODEL_VERSION = "resnet18-onnx-v1"
BENTO_MODEL = bentoml.models.get(f"{BENTO_MODEL_NAME}:latest")
onnx_runner = BENTO_MODEL.to_runner()
svc = bentoml.legacy.Service(
    "pet-breed-service",
    runners=[onnx_runner],
)


class PredictRequest(BaseModel):
    image: str


def predict_batch(
    requests: list[PredictRequest],
    runner,
) -> list[dict]:
    tensors = []

    for request in requests:
        image_bytes = base64.b64decode(
            request.image,
            validate=True,
        )

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                tensor = eval_transform(image.convert("RGB"))
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Request image is invalid.") from exc

        tensors.append(tensor.numpy())

    batch = np.stack(tensors).astype(np.float32)
    logits_batch = np.asarray(runner.run.run(batch))
    class_names = load_class_names()
    class_species = load_class_species()
    temperature = load_temperature()
    results = []

    for logits in logits_batch:
        scaled_logits = logits / temperature
        shifted = scaled_logits - np.max(scaled_logits)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        top_class_ids = np.argsort(probabilities)[::-1][:3]
        class_id = int(top_class_ids[0])
        confidence = float(probabilities[class_id])
        is_confident = confidence >= DEFAULT_CONFIDENCE_THRESHOLD

        results.append(
            {
                "breed": class_names[class_id] if is_confident else None,
                "species": class_species[class_id] if is_confident else None,
                "confidence": confidence,
                "top_3": [
                    {
                        "breed": class_names[int(index)],
                        "species": class_species[int(index)],
                        "confidence": float(probabilities[index]),
                    }
                    for index in top_class_ids
                ],
                "decision": "confident" if is_confident else "uncertain",
                "model_version": MODEL_VERSION,
                "threshold": DEFAULT_CONFIDENCE_THRESHOLD,
                "class_id": class_id,
            }
        )

    return results


@svc.api(
    input=bentoml.io.JSON(),
    output=bentoml.io.JSON(),
)
def predict(request: dict) -> dict:
    validated_request = PredictRequest.model_validate(request)
    return predict_batch([validated_request], onnx_runner)[0]
