from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image
from torchvision.datasets import OxfordIIITPet


SEED = 42


def apply_context_drift(
    image: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    """
    Apply realistic context/capture drift.

    The goal is to simulate real user-submitted images where:
    - the animal is farther from the camera
    - the animal is off-center
    - the framing is different
    - part of the animal may be cropped
    - another object may partially hide the animal
    - the camera angle may be different
    """

    original_height, original_width = image.shape[:2]

    # ==================================================
    # 1. Subject scale drift
    # ==================================================
    # Make the animal occupy a smaller portion of the
    # final image, simulating a photo taken from farther away.
    # ==================================================

    if rng.random() < 0.60:

        scale = rng.uniform(0.55, 0.85)

        new_width = max(
            1,
            int(original_width * scale),
        )

        new_height = max(
            1,
            int(original_height * scale),
        )

        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

        # Put the smaller image on a canvas.
        canvas = np.zeros_like(image)

        # Use a random background color based on the
        # border of the original image.
        border_pixels = np.concatenate(
            [
                image[0, :, :],
                image[-1, :, :],
                image[:, 0, :],
                image[:, -1, :],
            ],
            axis=0,
        )

        background_color = np.median(
            border_pixels,
            axis=0,
        ).astype(np.uint8)

        canvas[:] = background_color

        # Random position inside the canvas.
        max_top = original_height - new_height
        max_left = original_width - new_width

        top = rng.randint(
            0,
            max_top,
        )

        left = rng.randint(
            0,
            max_left,
        )

        canvas[
            top:top + new_height,
            left:left + new_width,
        ] = resized

        image = canvas

    # ==================================================
    # 2. Off-center subject
    # ==================================================
    # Shift the image slightly so the animal is no longer
    # necessarily centered.
    # ==================================================

    if rng.random() < 0.50:

        height, width = image.shape[:2]

        shift_x = rng.randint(
            int(-0.15 * width),
            int(0.15 * width),
        )

        shift_y = rng.randint(
            int(-0.10 * height),
            int(0.10 * height),
        )

        matrix = np.float32([
            [1, 0, shift_x],
            [0, 1, shift_y],
        ])

        image = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            borderMode=cv2.BORDER_REFLECT,
        )

    # ==================================================
    # 3. Partial framing / crop
    # ==================================================
    # Simulates a user taking a photo where part of the
    # animal may be close to the edge of the frame.
    # ==================================================

    if rng.random() < 0.35:

        height, width = image.shape[:2]

        crop_ratio = rng.uniform(
            0.80,
            0.95,
        )

        crop_width = int(
            width * crop_ratio
        )

        crop_height = int(
            height * crop_ratio
        )

        max_left = width - crop_width
        max_top = height - crop_height

        left = rng.randint(
            0,
            max_left,
        )

        top = rng.randint(
            0,
            max_top,
        )

        image = image[
            top:top + crop_height,
            left:left + crop_width,
        ]

        image = cv2.resize(
            image,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

    # ==================================================
    # 4. Partial occlusion
    # ==================================================
    # Simulates an object being in front of the animal:
    # furniture, hand, object, etc.
    #
    # We keep the occlusion small so that the image
    # remains realistic.
    # ==================================================

    if rng.random() < 0.25:

        height, width = image.shape[:2]

        occlusion_width = int(
            width * rng.uniform(
                0.08,
                0.20,
            )
        )

        occlusion_height = int(
            height * rng.uniform(
                0.08,
                0.20,
            )
        )

        x = rng.randint(
            0,
            width - occlusion_width,
        )

        y = rng.randint(
            0,
            height - occlusion_height,
        )

        # Sample the occlusion color from an existing
        # image region instead of using an unnatural
        # pure black rectangle.
        sample_x = rng.randint(
            0,
            max(0, width - occlusion_width),
        )

        sample_y = rng.randint(
            0,
            max(0, height - occlusion_height),
        )

        patch = image[
            sample_y:sample_y + occlusion_height,
            sample_x:sample_x + occlusion_width,
        ].copy()

        image[
            y:y + occlusion_height,
            x:x + occlusion_width,
        ] = patch

    # ==================================================
    # 5. Camera perspective
    # ==================================================
    # Simulates taking the picture from a slightly different
    # angle rather than directly facing the animal.
    # ==================================================

    if rng.random() < 0.30:

        height, width = image.shape[:2]

        shift = 0.08

        source_points = np.float32([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ])

        destination_points = np.float32([
            [
                rng.uniform(0, width * shift),
                rng.uniform(0, height * shift),
            ],
            [
                width - 1 - rng.uniform(
                    0,
                    width * shift,
                ),
                rng.uniform(
                    0,
                    height * shift,
                ),
            ],
            [
                width - 1 - rng.uniform(
                    0,
                    width * shift,
                ),
                height - 1 - rng.uniform(
                    0,
                    height * shift,
                ),
            ],
            [
                rng.uniform(
                    0,
                    width * shift,
                ),
                height - 1 - rng.uniform(
                    0,
                    height * shift,
                ),
            ],
        ])

        matrix = cv2.getPerspectiveTransform(
            source_points,
            destination_points,
        )

        image = cv2.warpPerspective(
            image,
            matrix,
            (width, height),
            borderMode=cv2.BORDER_REFLECT,
        )

    return np.clip(
        image,
        0,
        255,
    ).astype(np.uint8)


def generate_dataset() -> None:

    project_root = Path(__file__).resolve().parents[1]

    raw_root = project_root / "data" / "raw"

    output_root = (
        project_root
        / "data"
        / "processed"
        / "v3_context_drift"
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
        f"Generating V3 context drift dataset: "
        f"{len(dataset)} images"
    )

    for index in range(len(dataset)):

        image, label = dataset[index]

        image = np.array(
            image.convert("RGB")
        )

        drifted_image = apply_context_drift(
            image,
            rng,
        )

        class_name = dataset.classes[label]

        class_dir = (
            output_root / class_name
        )

        class_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            class_dir / f"{index:05d}.jpg"
        )

        Image.fromarray(
            drifted_image
        ).save(
            output_path,
            quality=90,
        )

    print(
        "V3 context drift dataset generation completed."
    )

    print(
        f"Output directory: {output_root}"
    )


if __name__ == "__main__":
    generate_dataset()