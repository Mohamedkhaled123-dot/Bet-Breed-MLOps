from pathlib import Path

from PIL import Image
from torchvision.datasets import OxfordIIITPet


EXPECTED_CLASSES = 37

EXPECTED_SPLIT_COUNTS = {
    "train": 2944,
    "val": 736,
    "test": 3669,
}


def count_images(directory: Path) -> int:
    """Count image files inside a directory recursively."""

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
    }

    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in image_extensions
    )


def get_classes(directory: Path) -> set[str]:
    """Return class folder names."""

    return {
        path.name
        for path in directory.iterdir()
        if path.is_dir()
    }


def validate_image_integrity(directory: Path) -> int:
    """Check that all images can be opened correctly."""

    corrupted_images = []

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
    }

    for image_path in directory.rglob("*"):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in image_extensions:
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()

        except Exception:
            corrupted_images.append(image_path)

    return len(corrupted_images)


def validate_split_dataset(
    dataset_root: Path,
    dataset_name: str,
) -> None:

    print(f"\n{'=' * 50}")
    print(f"Validating {dataset_name}")
    print(f"{'=' * 50}")

    expected_classes = None

    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():

        split_dir = dataset_root / split

        assert split_dir.exists(), (
            f"Missing directory: {split_dir}"
        )

        classes = get_classes(split_dir)

        image_count = count_images(split_dir)

        print(
            f"{split:5} | "
            f"images: {image_count:4} | "
            f"classes: {len(classes):2}"
        )

        # Check image count
        assert image_count == expected_count, (
            f"{dataset_name} {split}: "
            f"expected {expected_count}, "
            f"found {image_count}"
        )

        # Check number of classes
        assert len(classes) == EXPECTED_CLASSES, (
            f"{dataset_name} {split}: "
            f"expected {EXPECTED_CLASSES} classes, "
            f"found {len(classes)}"
        )

        # Make sure every split has the same classes
        if expected_classes is None:
            expected_classes = classes
        else:
            assert classes == expected_classes, (
                f"{dataset_name}: class mismatch in {split}"
            )

        # Check image integrity
        corrupted_count = validate_image_integrity(
            split_dir
        )

        print(
            f"{'':5} | "
            f"corrupted: {corrupted_count}"
        )

        assert corrupted_count == 0, (
            f"{dataset_name} {split} contains "
            f"{corrupted_count} corrupted images"
        )

    print(
        f"\n{dataset_name} validation PASSED."
    )


def validate_raw_dataset(
    project_root: Path,
) -> None:

    print("\n" + "=" * 50)
    print("Validating RAW Oxford-IIIT Pet dataset")
    print("=" * 50)

    data_root = project_root / "data" / "raw"

    print("\nLoading dataset metadata...")

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

    print(
        f"Trainval samples : {len(train_dataset)}"
    )

    print(
        f"Test samples     : {len(test_dataset)}"
    )

    print(
        f"Classes          : {len(train_dataset.classes)}"
    )

    assert len(train_dataset) == 3680
    assert len(test_dataset) == 3669
    assert len(train_dataset.classes) == EXPECTED_CLASSES

    image_dir = (
        data_root
        / "oxford-iiit-pet"
        / "images"
    )

    image_files = list(
        image_dir.glob("*.jpg")
    )

    print(
        f"\nRaw image files found: "
        f"{len(image_files)}"
    )

    assert len(image_files) > 0

    corrupted_count = validate_image_integrity(
        image_dir
    )

    print(
        f"Corrupted images: {corrupted_count}"
    )

    assert corrupted_count == 0

    print("\nRAW validation PASSED.")


def validate_dataset() -> None:

    project_root = Path(__file__).resolve().parents[1]

    # ---------------------------------------------
    # 1. RAW
    # ---------------------------------------------

    validate_raw_dataset(project_root)

    # ---------------------------------------------
    # 2. SPLIT
    # ---------------------------------------------

    split_root = (
        project_root
        / "data"
        / "raw"
        / "split"
    )

    validate_split_dataset(
        split_root,
        "RAW SPLIT",
    )

    # ---------------------------------------------
    # 3. V2
    # ---------------------------------------------

    v2_root = (
        project_root
        / "data"
        / "processed"
        / "v2_realistic"
    )

    validate_split_dataset(
        v2_root,
        "V2 REALISTIC",
    )

    # ---------------------------------------------
    # 4. V3
    # ---------------------------------------------

    v3_root = (
        project_root
        / "data"
        / "processed"
        / "v3_context_drift"
    )

    validate_split_dataset(
        v3_root,
        "V3 CONTEXT DRIFT",
    )

    # ---------------------------------------------
    # DONE
    # ---------------------------------------------

    print("\n" + "=" * 50)
    print("ALL DATASET VALIDATION PASSED.")
    print("=" * 50)


if __name__ == "__main__":
    validate_dataset()