from pathlib import Path

from PIL import Image

from bet_breed_mlops.corruptions import CORRUPTIONS


def test_corruptions():

    project_root = Path(__file__).resolve().parents[1]

    test_dir = (
        project_root
        / "data"
        / "raw"
        / "split"
        / "test"
    )

    image_paths = [
        path
        for path in test_dir.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    assert len(image_paths) > 0

    image_path = image_paths[0]

    with Image.open(image_path) as image:

        image = image.convert("RGB")

        original_size = image.size

        for corruption_name, corruption_function in CORRUPTIONS.items():

            for severity in range(1, 4):

                corrupted = corruption_function(
                    image,
                    severity,
                )

                assert corrupted.mode == "RGB"

                assert corrupted.width > 0
                assert corrupted.height > 0

                assert corrupted.size == original_size