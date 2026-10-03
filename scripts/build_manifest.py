from pathlib import Path
import json

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

SPLITS = ["train", "val", "test"]


def build_manifest() -> None:

    project_root = Path(__file__).resolve().parents[1]

    data_root = project_root / "data" / "raw" / "split"

    output_path = (
        project_root
        / "data"
        / "metadata"
        / "manifest.json"
    )

    records = []

    # --------------------------------------------------------
    # Build fixed label map from all 37 classes
    # --------------------------------------------------------

    classes = sorted(
        {
            class_dir.name
            for split in SPLITS
            for class_dir in (data_root / split).iterdir()
            if class_dir.is_dir()
        }
    )

    label_map = {
        class_name: index
        for index, class_name in enumerate(classes)
    }

    # --------------------------------------------------------
    # Read images
    # --------------------------------------------------------

    for split in SPLITS:

        split_dir = data_root / split

        for image_path in sorted(split_dir.rglob("*")):

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            breed = image_path.parent.name

            image_id = image_path.stem

            with Image.open(image_path) as image:

                width, height = image.size

            species = (
                "cat"
                if breed.endswith("Cat")
                or breed in {
                    "Abyssinian",
                    "Bengal",
                    "Birman",
                    "Bombay",
                    "British Shorthair",
                    "Egyptian Mau",
                    "Maine Coon",
                    "Persian",
                    "Ragdoll",
                    "Russian Blue",
                    "Siamese",
                    "Sphynx",
                }
                else "dog"
            )

            record = {
                "image_id": image_id,
                "path": image_path.relative_to(
                    project_root
                ).as_posix(),
                "breed": breed,
                "species": species,
                "class_index": label_map[breed],
                "split": split,
                "corruption": None,
                "severity": 0,
                "width": width,
                "height": height,
            }

            records.append(record)

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "label_map": label_map,
        "images": records,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Manifest created: {output_path}"
    )

    print(
        f"Classes: {len(classes)}"
    )

    print(
        f"Images: {len(records)}"
    )


if __name__ == "__main__":
    build_manifest()