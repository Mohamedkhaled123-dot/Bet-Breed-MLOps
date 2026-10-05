from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from bet_breed_mlops.onnx_inference import ONNXInference


app = FastAPI(
    title="Pet Breed Prediction API",
    version="0.1.0",
)


inference = ONNXInference()


@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
):
    try:
        image_bytes = await file.read()

        image = Image.open(
            __import__("io").BytesIO(image_bytes)
        ).convert("RGB")

    except (
        UnidentifiedImageError,
        OSError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file.",
        ) from exc

    result = inference.predict(image)

    return {
        "filename": file.filename,
        **result,
    }