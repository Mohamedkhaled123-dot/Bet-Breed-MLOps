
import os
import time
import warnings
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Literal, Protocol

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from bet_breed_mlops.onnx_inference import ONNXInference
from bet_breed_mlops.prometheus_metrics import (
    observe_prediction,
    render_prometheus_metrics,
)

MODEL_VERSION = os.getenv(
    "MODEL_VERSION",
    "resnet18-onnx-cpu-v1",
)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
NUM_CLASSES = 37


class BreedPrediction(BaseModel):
    breed: str
    species: Literal["cat", "dog"]
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    filename: str | None
    breed: str | None
    species: Literal["cat", "dog"] | None
    confidence: float = Field(ge=0.0, le=1.0)
    top_3: list[BreedPrediction]
    decision: Literal["confident", "uncertain"]
    model_version: str
    threshold: float = Field(ge=0.0, le=1.0)
    status: Literal["Confident", "Not sure"]
    class_id: int = Field(ge=0, lt=NUM_CLASSES)


class InferenceEngine(Protocol):
    class_names: dict[int, str]

    def predict(self, image: Image.Image) -> dict: ...


def create_app(
    inference_factory=ONNXInference,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.inference = inference_factory()
        yield

    api = FastAPI(
        title="Pet Breed Prediction API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @api.get("/health")
    def health_check():
        inference: InferenceEngine = api.state.inference
        return {
            "status": "healthy",
            "model_version": MODEL_VERSION,
            "num_classes": len(inference.class_names),
        }

    @api.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        content, content_type = render_prometheus_metrics()
        return Response(content=content, media_type=content_type)

    @api.post(
        "/predict",
        response_model=PredictionResponse,
    )
    async def predict(
        file: UploadFile = File(...),
    ) -> PredictionResponse:
        image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)

        if len(image_bytes) > MAX_UPLOAD_BYTES:
            await file.close()
            raise HTTPException(
                status_code=413,
                detail=f"Image upload exceeds {MAX_UPLOAD_BYTES} bytes.",
            )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "error",
                    Image.DecompressionBombWarning,
                )
                with Image.open(BytesIO(image_bytes)) as uploaded_image:
                    uploaded_image.load()
                    image = uploaded_image.convert("RGB")
        except (
            UnidentifiedImageError,
            OSError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as exc:
            raise HTTPException(
                status_code=422,
                detail="Upload must contain a valid, decodable image.",
            ) from exc
        finally:
            await file.close()

        inference: InferenceEngine = api.state.inference
        started_at = time.perf_counter()

        try:
            result = inference.predict(image)
        finally:
            observe_prediction(time.perf_counter() - started_at)

        return PredictionResponse(
            filename=file.filename,
            model_version=MODEL_VERSION,
            **result,
        )

    return api


app = create_app()
