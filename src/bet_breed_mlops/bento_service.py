import base64
from io import BytesIO
from pathlib import Path

import bentoml
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from pydantic import BaseModel

from bet_breed_mlops.confidence import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    load_class_names,
    load_temperature,
    predict_with_confidence,
)
from bet_breed_mlops.data import eval_transform


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = PROJECT_ROOT / "artifacts" / "models" / "resnet18.onnx"


class PredictRequest(BaseModel):
    image: str


@bentoml.service
class PetBreedService:

    def __init__(self):
        self.session = ort.InferenceSession(
            str(ONNX_PATH),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.temperature = load_temperature()
        self.class_names = load_class_names()

    @bentoml.api(
        batchable=True,
        max_batch_size=32,
        max_latency_ms=100,
    )
    def predict(
        self,
        requests: list[PredictRequest],
    ) -> list[dict]:

        tensors = []

        for request in requests:
            image_bytes = base64.b64decode(request.image)

            image = Image.open(
                BytesIO(image_bytes)
            ).convert("RGB")

            tensor = eval_transform(image)
            tensors.append(tensor.numpy())

        batch = np.stack(tensors).astype(np.float32)

        logits_batch = self.session.run(
            None,
            {self.input_name: batch},
        )[0]

        results = []

        for logits in logits_batch:
            result = predict_with_confidence(
                logits=torch.from_numpy(logits),
                temperature=self.temperature,
                threshold=DEFAULT_CONFIDENCE_THRESHOLD,
                class_names=self.class_names,
            )

            results.append(
                {
                    "class_index": result.class_index,
                    "breed": result.breed,
                    "confidence": result.confidence,
                    "is_confident": result.is_confident,
                }
            )

        return results