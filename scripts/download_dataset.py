
from pathlib import Path

from torchvision.datasets import OxfordIIITPet


def download_dataset() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "raw"

    data_root.mkdir(parents=True, exist_ok=True)

    print("Downloading Oxford-IIIT Pet dataset...")

    train_dataset = OxfordIIITPet(
        root=data_root,
        split="trainval",
        target_types="category",
        download=True,
    )

    test_dataset = OxfordIIITPet(
        root=data_root,
        split="test",
        target_types="category",
        download=True,
    )

    print(f"Trainval samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print("Dataset download completed.")


if __name__ == "__main__":
    download_dataset()