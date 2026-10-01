from pathlib import Path

from bet_breed_mlops.data import get_dataloaders


def test_dataloader():

    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "raw" / "split"

    train_loader, val_loader, test_loader = get_dataloaders(
        data_root
    )

    images, labels = next(iter(train_loader))

    # Check image shape
    assert images.shape[0] <= 32
    assert images.shape[1] == 3
    assert images.shape[2] == 224
    assert images.shape[3] == 224

    # Check labels
    assert len(images) == len(labels)

    # Check number of classes
    assert len(train_loader.dataset.classes) == 37
    assert len(val_loader.dataset.classes) == 37
    assert len(test_loader.dataset.classes) == 37


def test_dataset_info():

    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "raw" / "split"

    train_loader, val_loader, test_loader = get_dataloaders(
        data_root
    )

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset

    print("\nTrain images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Test images:", len(test_dataset))

    print("\nNumber of classes:", len(train_dataset.classes))

    print("\nClasses:")
    for idx, class_name in enumerate(train_dataset.classes):
        print(idx, "->", class_name)
from pathlib import Path

from bet_breed_mlops.data import get_dataloaders


def test_dataloader():

    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "raw" / "split"

    train_loader, val_loader, test_loader = get_dataloaders(
        data_root
    )

    images, labels = next(iter(train_loader))

    # Check image shape
    assert images.shape[0] <= 32
    assert images.shape[1] == 3
    assert images.shape[2] == 224
    assert images.shape[3] == 224

    # Check labels
    assert len(images) == len(labels)

    # Check number of classes
    assert len(train_loader.dataset.classes) == 37
    assert len(val_loader.dataset.classes) == 37
    assert len(test_loader.dataset.classes) == 37


def test_dataset_info():

    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "raw" / "split"

    train_loader, val_loader, test_loader = get_dataloaders(
        data_root
    )

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset

    print("\nTrain images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Test images:", len(test_dataset))

    print("\nNumber of classes:", len(train_dataset.classes))

    print("\nClasses:")
    for idx, class_name in enumerate(train_dataset.classes):
        print(idx, "->", class_name)

from pathlib import Path

from bet_breed_mlops.data import get_dataloaders


def test_dataloader():

    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "raw" / "split"

    train_loader, val_loader, test_loader = get_dataloaders(
        data_root
    )

    images, labels = next(iter(train_loader))

    # Check image shape
    assert images.shape[0] <= 32
    assert images.shape[1] == 3
    assert images.shape[2] == 224
    assert images.shape[3] == 224

    # Check labels
    assert len(images) == len(labels)

    # Check number of classes
    assert len(train_loader.dataset.classes) == 37
    assert len(val_loader.dataset.classes) == 37
    assert len(test_loader.dataset.classes) == 37


def test_dataset_info():

    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "raw" / "split"

    train_loader, val_loader, test_loader = get_dataloaders(
        data_root
    )

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset

    print("\nTrain images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Test images:", len(test_dataset))

    print("\nNumber of classes:", len(train_dataset.classes))

    print("\nClasses:")
    for idx, class_name in enumerate(train_dataset.classes):
        print(idx, "->", class_name)