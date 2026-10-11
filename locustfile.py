import base64
import os
from pathlib import Path

from locust import HttpUser, between, task

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "split" / "test"
LOAD_MODE = os.getenv("PET_BREED_LOAD_MODE", "fastapi").lower()


class PetBreedInferenceUser(HttpUser):
    wait_time = between(0.05, 0.2)

    def on_start(self) -> None:
        if LOAD_MODE not in {"fastapi", "bentoml"}:
            raise ValueError(
                "PET_BREED_LOAD_MODE must be 'fastapi' or 'bentoml'."
            )
        sample_images = sorted(
            path
            for path in IMAGE_ROOT.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not sample_images:
            raise FileNotFoundError(f"No test images found under {IMAGE_ROOT}.")
        sample_image = sample_images[0]
        self.image_name = sample_image.name
        self.content_type = (
            "image/png"
            if sample_image.suffix.lower() == ".png"
            else "image/jpeg"
        )
        self.image_bytes = sample_image.read_bytes()

    @task
    def predict(self) -> None:
        with self.client.post(
            "/predict",
            json={
                "image": base64.b64encode(self.image_bytes).decode("ascii"),
            }
            if LOAD_MODE == "bentoml"
            else None,
            files=(
                None
                if LOAD_MODE == "bentoml"
                else {
                    "file": (
                        self.image_name,
                        self.image_bytes,
                        self.content_type,
                    )
                }
            ),
            name=f"/predict ({LOAD_MODE})",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Unexpected HTTP status {response.status_code}.")
                return
            result = response.json()
            if "top_3" not in result or "decision" not in result:
                response.failure("Prediction response does not match the API contract.")
