from pathlib import Path
import io

from PIL import Image, ImageEnhance, ImageFilter


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ============================================================
# 1. Gaussian Blur
# ============================================================

def gaussian_blur(
    image: Image.Image,
    severity: int,
) -> Image.Image:

    radius = {
        1: 1.0,
        2: 2.0,
        3: 3.0,
    }[severity]

    return image.filter(
        ImageFilter.GaussianBlur(radius=radius)
    )


# ============================================================
# 2. Brightness
# ============================================================

def brightness_shift(
    image: Image.Image,
    severity: int,
    direction: str,
) -> Image.Image:

    factors = {
        1: 0.8,
        2: 0.6,
        3: 0.4,
    }

    if direction == "down":
        factor = factors[severity]

    elif direction == "up":
        factor = 1 / factors[severity]

    else:
        raise ValueError(
            "direction must be 'up' or 'down'"
        )

    return ImageEnhance.Brightness(
        image
    ).enhance(factor)


def brightness_down(
    image: Image.Image,
    severity: int,
) -> Image.Image:

    return brightness_shift(
        image,
        severity,
        direction="down",
    )


def brightness_up(
    image: Image.Image,
    severity: int,
) -> Image.Image:

    return brightness_shift(
        image,
        severity,
        direction="up",
    )


# ============================================================
# 3. JPEG Compression
# ============================================================

def jpeg_compression(
    image: Image.Image,
    severity: int,
) -> Image.Image:

    qualities = {
        1: 60,
        2: 45,
        3: 30,
    }

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=qualities[severity],
    )

    buffer.seek(0)

    return Image.open(buffer).convert("RGB").copy()


# ============================================================
# 4. Downscale → Upscale
# ============================================================

def low_resolution(
    image: Image.Image,
    severity: int,
) -> Image.Image:

    sizes = {
        1: 192,
        2: 128,
        3: 96,
    }

    size = sizes[severity]

    small = image.resize(
        (size, size),
        Image.Resampling.BILINEAR,
    )

    return small.resize(
        image.size,
        Image.Resampling.BILINEAR,
    )


# ============================================================
# 5. Motion Blur
# ============================================================

def motion_blur(
    image: Image.Image,
    severity: int,
) -> Image.Image:

    kernel_sizes = {
        1: 3,
        2: 5,
        3: 5,
    }

    kernel_size = kernel_sizes[severity]

    kernel = [1.0 / kernel_size] * kernel_size

    full_kernel = []

    for row in range(kernel_size):

        if row == kernel_size // 2:
            full_kernel.extend(kernel)

        else:
            full_kernel.extend(
                [0.0] * kernel_size
            )

    result = image.filter(
        ImageFilter.Kernel(
            size=(kernel_size, kernel_size),
            kernel=full_kernel,
            scale=1.0,
        )
    )

    if severity == 3:
        result = result.filter(
            ImageFilter.Kernel(
                size=(kernel_size, kernel_size),
                kernel=full_kernel,
                scale=1.0,
            )
        )

    return result


# ============================================================
# Corruption Registry
# ============================================================

CORRUPTIONS = {
    "gaussian_blur": gaussian_blur,
    "brightness_down": brightness_down,
    "brightness_up": brightness_up,
    "jpeg": jpeg_compression,
    "low_resolution": low_resolution,
    "motion_blur": motion_blur,
}


# ============================================================
# Dataset Generator
# ============================================================

def generate_corrupted_dataset(
    input_dir: Path,
    output_dir: Path,
) -> None:

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for corruption_name, corruption_function in CORRUPTIONS.items():

        for severity in range(1, 4):

            corruption_output = (
                output_dir
                / corruption_name
                / f"severity_{severity}"
            )

            corruption_output.mkdir(
                parents=True,
                exist_ok=True,
            )

            for image_path in input_dir.rglob("*"):

                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                relative_path = image_path.relative_to(
                    input_dir
                )

                output_path = (
                    corruption_output
                    / relative_path
                )

                output_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                # JPEG corruption should be saved as JPG
                if corruption_name == "jpeg":
                    output_path = output_path.with_suffix(".jpg")

                with Image.open(image_path) as image:

                    # Important:
                    # All images become RGB
                    image = image.convert("RGB")

                    corrupted = corruption_function(
                        image,
                        severity,
                    )

                    if corruption_name == "jpeg":
                        corrupted.save(
                            output_path,
                            format="JPEG",
                        )
                    else:
                        corrupted.save(
                            output_path
                        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[2]

    test_dir = (
        project_root
        / "data"
        / "raw"
        / "split"
        / "test"
    )

    output_dir = (
        project_root
        / "data"
        / "processed"
        / "corrupted_test"
    )

    generate_corrupted_dataset(
        test_dir,
        output_dir,
    )

    print (project_root)
    print(
        "Corrupted dataset generated successfully."
    )