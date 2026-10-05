from pathlib import Path
import csv
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

from bet_breed_mlops.confidence import load_class_names, load_temperature
from bet_breed_mlops.data import eval_transform


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ONNX_PATH = PROJECT_ROOT / "artifacts" / "models" / "resnet18.onnx"
TEST_DIR = PROJECT_ROOT / "data" / "raw" / "split" / "test"
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "batch_predictions.csv"

NUM_IMAGES = 1000
BATCH_SIZE = 32


def collect_images():
    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    images = sorted(
        path
        for path in TEST_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )

    if len(images) < NUM_IMAGES:
        raise RuntimeError(
            f"Need at least {NUM_IMAGES} images, found {len(images)}."
        )

    return images[:NUM_IMAGES]


def load_batch(paths):
    tensors = []

    for path in paths:
        image = Image.open(path).convert("RGB")
        tensor = eval_transform(image)
        tensors.append(tensor.numpy())

    return np.stack(tensors).astype(np.float32)


def main():
    image_paths = collect_images()

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    class_names = load_class_names()
    temperature = load_temperature()

    predictions = []

    start = time.perf_counter()

    for start_idx in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[start_idx : start_idx + BATCH_SIZE]

        batch = load_batch(batch_paths)

        logits = session.run(
            None,
            {input_name: batch},
        )[0]

        calibrated_logits = logits / temperature

        shifted = calibrated_logits - np.max(
            calibrated_logits,
            axis=1,
            keepdims=True,
        )

        probabilities = np.exp(shifted)
        probabilities /= np.sum(
            probabilities,
            axis=1,
            keepdims=True,
        )

        predicted_classes = np.argmax(
            probabilities,
            axis=1,
        )

        confidences = np.max(
            probabilities,
            axis=1,
        )

        for path, class_id, confidence in zip(
            batch_paths,
            predicted_classes,
            confidences,
        ):
            predictions.append(
                {
                    "image_path": str(path.relative_to(PROJECT_ROOT)),
                    "predicted_class": class_names[int(class_id)],
                    "confidence": float(confidence),
                }
            )

    elapsed = time.perf_counter() - start

    throughput = len(predictions) / elapsed

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_path",
                "predicted_class",
                "confidence",
            ],
        )

        writer.writeheader()
        writer.writerows(predictions)

    print("\nBatch Scoring Results")
    print("=====================")
    print(f"Images scored : {len(predictions)}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Total time    : {elapsed:.3f} sec")
    print(f"Throughput    : {throughput:.2f} images/sec")
    print(f"Output CSV    : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()