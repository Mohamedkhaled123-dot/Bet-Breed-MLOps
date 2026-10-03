import json
from collections import Counter
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "manifest.json"
)


def load_manifest():
    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def test_manifest_exists():
    assert MANIFEST_PATH.exists()


def test_all_paths_exist_and_open():
    manifest = load_manifest()

    for record in manifest["images"]:
        image_path = PROJECT_ROOT / record["path"]

        assert image_path.exists(), (
            f"Missing image: {image_path}"
        )

        with Image.open(image_path) as image:
            image.verify()


def test_no_image_id_in_multiple_splits():
    manifest = load_manifest()

    image_splits = {}

    for record in manifest["images"]:
        image_id = record["image_id"]
        split = record["split"]

        if image_id in image_splits:
            assert image_splits[image_id] == split, (
                f"{image_id} appears in multiple splits"
            )

        image_splits[image_id] = split


def test_label_map_has_37_classes():
    manifest = load_manifest()

    label_map = manifest["label_map"]

    assert len(label_map) == 37

    assert set(label_map.values()) == set(range(37))


def test_every_class_has_at_least_50_training_images():
    manifest = load_manifest()

    train_records = [
        record
        for record in manifest["images"]
        if record["split"] == "train"
    ]

    class_counts = Counter(
        record["breed"]
        for record in train_records
    )

    assert len(class_counts) == 37

    for breed, count in class_counts.items():
        assert count >= 50, (
            f"{breed} has only {count} training images"
        )


def test_no_image_smaller_than_32x32():
    manifest = load_manifest()

    for record in manifest["images"]:

        width = record["width"]
        height = record["height"]

        assert width >= 32, (
            f"{record['image_id']} width is {width}"
        )

        assert height >= 32, (
            f"{record['image_id']} height is {height}"
        )