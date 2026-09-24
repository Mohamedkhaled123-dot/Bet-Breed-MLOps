from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image
from torchvision.datasets import OxfordIIITPet


SEED = 42


def apply_realistic_effects(
    image: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    
    """
    Apply realistic smartphone-like image degradations
    to simulate real-world camera/data drift.
    """
    image = np.asarray(image, dtype=np.uint8)
    # --------------------------------------------------
    # 1. Brightness + Contrast variation
    # --------------------------------------------------
    if rng.random() < 0.40:

        brightness = rng.uniform(0.85, 1.15)

        image = image.astype(np.float32) * brightness

        contrast = rng.uniform(0.85, 1.15)

        mean = image.mean()

        image = (image - mean) * contrast + mean

        image = np.clip(image, 0, 255)

    # --------------------------------------------------
    # 2. Color variation
    # --------------------------------------------------
    if rng.random() < 0.30:

        hsv = cv2.cvtColor(
            image.astype(np.uint8),
            cv2.COLOR_RGB2HSV,
        )

        # Hue
        hue_shift = rng.randint(-8, 8)

        hsv[:, :, 0] = (
            hsv[:, :, 0].astype(np.int16)
            + hue_shift
        ) % 180

        # Saturation
        saturation_scale = rng.uniform(0.88, 1.12)

        hsv[:, :, 1] = np.clip(
            hsv[:, :, 1].astype(np.float32)
            * saturation_scale,
            0,
            255,
        )

        # Brightness / Value
        value_scale = rng.uniform(0.92, 1.08)

        hsv[:, :, 2] = np.clip(
            hsv[:, :, 2].astype(np.float32)
            * value_scale,
            0,
            255,
        )

        image = cv2.cvtColor(
            hsv.astype(np.uint8),
            cv2.COLOR_HSV2RGB,
        )

    # --------------------------------------------------
    # 3. Blur
    # --------------------------------------------------
    if rng.random() < 0.20:

        if rng.random() < 0.5:

            # Gaussian blur
            kernel_size = rng.choice([3, 5])

            image = cv2.GaussianBlur(
                image,
                (kernel_size, kernel_size),
                0,
            )

        else:

            # Motion-like blur
            kernel_size = rng.choice([3, 5])

            kernel = np.zeros(
                (kernel_size, kernel_size)
            )

            kernel[kernel_size // 2, :] = (
                1.0 / kernel_size
            )

            image = cv2.filter2D(
                image,
                -1,
                kernel,
            )

    # --------------------------------------------------
    # 4. Gaussian noise
    # --------------------------------------------------
    if rng.random() < 0.15:

        noise_rng = np.random.default_rng(
            rng.randint(0, 1_000_000)
        )

        noise = noise_rng.normal(
            0,
            3,
            image.shape,
        )

        image = np.clip(
            image.astype(np.float32) + noise,
            0,
            255,
        )

    # --------------------------------------------------
    # 5. Small rotation
    # --------------------------------------------------
    if rng.random() < 0.15:

        angle = rng.uniform(-10, 10)

        height, width = image.shape[:2]

        center = (
            width / 2,
            height / 2,
        )

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0,
        )

        image = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            borderMode=cv2.BORDER_REFLECT,
        )

    # --------------------------------------------------
    # 6. Small crop / zoom
    # --------------------------------------------------
    if rng.random() < 0.20:

        height, width = image.shape[:2]

        scale = rng.uniform(0.85, 1.0)

        new_height = int(height * scale)
        new_width = int(width * scale)

        top = rng.randint(
            0,
            height - new_height,
        )

        left = rng.randint(
            0,
            width - new_width,
        )

        image = image[
            top:top + new_height,
            left:left + new_width,
        ]

        image = cv2.resize(
            image,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

    # --------------------------------------------------
    # 7. Resolution variation
    # --------------------------------------------------
    if rng.random() < 0.30:

        height, width = image.shape[:2]

        scale = rng.uniform(0.50, 0.75)

        small_width = max(1, int(width * scale))
        small_height = max(1, int(height * scale))

        # Reduce resolution
        image = cv2.resize(
            image,
            (small_width, small_height),
            interpolation=cv2.INTER_AREA,
        )

        # Resize back
        image = cv2.resize(
            image,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

    # --------------------------------------------------
    # 8. JPEG compression
    # --------------------------------------------------
    if rng.random() < 0.30:

        quality = rng.randint(70, 90)

        # Make sure OpenCV receives a valid uint8 image
        image = np.clip(
            image,
            0,
            255,
        ).astype(np.uint8)

        # RGB -> BGR
        bgr_image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR,
        )

        success, encoded = cv2.imencode(
            ".jpg",
            bgr_image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                quality,
            ],
        )

        if success:

            decoded = cv2.imdecode(
                encoded,
                cv2.IMREAD_COLOR,
            )

            image = cv2.cvtColor(
                decoded,
                cv2.COLOR_BGR2RGB,
            )
    return np.clip(
    image,
    0,
    255,).astype(np.uint8)

def generate_dataset() -> None:

    project_root = Path(__file__).resolve().parents[1]

    raw_root = project_root / "data" / "raw"

    output_root = (
        project_root
        / "data"
        / "processed"
        / "v2_realistic"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = OxfordIIITPet(
        root=raw_root,
        split="trainval",
        target_types="category",
        download=False,
    )

    rng = random.Random(SEED)

    print(
        f"Generating V2 realistic dataset: "
        f"{len(dataset)} images"
    )

    for index in range(len(dataset)):

        image, label = dataset[index]

        image = np.array(
            image.convert("RGB")
        )

        realistic_image = apply_realistic_effects(
            image,
            rng,
        )
        

        class_name = dataset.classes[label]

        class_dir = output_root / class_name

        class_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            class_dir / f"{index:05d}.jpg"
        )

        Image.fromarray(
            realistic_image
        ).save(
            output_path,
            quality=90,
        )

    print(
        "V2 realistic dataset generation completed."
    )

    print(
        f"Output directory: {output_root}"
    )


if __name__ == "__main__":
    generate_dataset()