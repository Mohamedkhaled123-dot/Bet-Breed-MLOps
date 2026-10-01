from pathlib import Path
import shutil

from torchvision.datasets import OxfordIIITPet
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
VAL_SIZE = 0.20


def split_dataset() -> None:
    project_root = Path(__file__).resolve().parents[1]

    data_root = project_root / "data" / "raw" 

    output_root = project_root / "data" / "raw" / "split"

    print("Loading Oxford-IIIT Pet dataset...")

    trainval_dataset = OxfordIIITPet(
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

    image_dir = data_root / "oxford-iiit-pet" / "images"

    print(f"Total trainval samples: {len(trainval_dataset)}")
    print(f"Total test samples    : {len(test_dataset)}")
    print(f"Number of classes      : {len(trainval_dataset.classes)}")

    # Create labels
    labels = [trainval_dataset[i][1] for i in range(len(trainval_dataset))]
    indices = list(range(len(trainval_dataset)))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples  : {len(val_indices)}")
    print(f"Test samples : {len(test_dataset)}")

    # Remove previous output if it exists
    if output_root.exists():
        print("\nRemoving previous split...")
        shutil.rmtree(output_root)

    train_dir = output_root / "train"
    val_dir = output_root / "val"
    test_dir = output_root / "test"

    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    def copy_images(
        dataset,
        indices: list[int],
        destination: Path,
    ) -> None:

        for idx in indices:
            image_path = Path(dataset._images[idx])

            class_name = dataset.classes[dataset[idx][1]]

            class_dir = destination / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(
                image_dir / image_path.name,
                class_dir / image_path.name,
            )

    # -----------------------------------------
    # 7. Copy train images
    # -----------------------------------------

    print("\nCopying training images...")
    copy_images(
        trainval_dataset,
        train_indices,
        train_dir,
    )

    # -----------------------------------------
    # 8. Copy validation images
    # -----------------------------------------

    print("Copying validation images...")
    copy_images(
        trainval_dataset,
        val_indices,
        val_dir,
    )

    # -----------------------------------------
    # 9. Copy test images
    # -----------------------------------------

    print("Copying test images...")

    test_indices = list(range(len(test_dataset)))

    copy_images(
        test_dataset,
        test_indices,
        test_dir,
    )

    # -----------------------------------------
    # 10. Done
    # -----------------------------------------

    print("\nDataset split completed.")

    print("\nOutput:")
    print(f"Train: {train_dir}")
    print(f"Val  : {val_dir}")
    print(f"Test : {test_dir}")


if __name__ == "__main__":
    split_dataset()
 