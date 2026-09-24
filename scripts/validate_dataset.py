from pathlib import Path

from PIL import Image
from torchvision.datasets import OxfordIIITPet


def validate_dataset() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "raw"

    print("Loading dataset metadata...")

    train_dataset = OxfordIIITPet(
        root=data_root,
        split="trainval",
        target_types="category",
        download=False,
    )

    test_dataset = OxfordIIITPet(
        root=data_root,
        split="test",
        target_types="category",
        download=False,
    )

    print(f"Trainval samples : {len(train_dataset)}")
    print(f"Test samples     : {len(test_dataset)}")
    print(f"Classes          : {len(train_dataset.classes)}")

    # Expected values
    assert len(train_dataset) == 3680
    assert len(test_dataset) == 3669
    assert len(train_dataset.classes) == 37

    print("\nChecking image files...")

    image_dir = data_root / "oxford-iiit-pet" / "images"

    image_files = list(image_dir.glob("*.jpg"))

    print(f"Image files found: {len(image_files)}")

    assert len(image_files) > 0

    print("\nChecking image integrity...")

    corrupted_images = []

    for image_path in image_files:
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception:
            corrupted_images.append(image_path)

    print(f"Corrupted images: {len(corrupted_images)}")

    assert len(corrupted_images) == 0

    print("\nDataset validation PASSED.")


if __name__ == "__main__":
    validate_dataset()