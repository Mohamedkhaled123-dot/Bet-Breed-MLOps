from bet_breed_mlops.data import create_dataloaders


def test_dataloader():

    train_loader, val_loader = create_dataloaders(
        batch_size=32
    )

    images, labels = next(iter(train_loader))

    # Check image shape
    assert images.shape[0] <= 32
    assert images.shape[1] == 3
    assert images.shape[2] == 224
    assert images.shape[3] == 224

    # Check labels
    assert len(images) == len(labels)


def test_dataset_sizes():

    train_loader, val_loader = create_dataloaders(
        batch_size=32
    )

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    print("\nTrain images:", len(train_dataset))
    print("Validation images:", len(val_dataset))

    assert len(train_dataset) == 2944
    assert len(val_dataset) == 736


def test_number_of_classes():

    train_loader, val_loader = create_dataloaders(
        batch_size=32
    )

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    train_classes = {
        record["class_index"]
        for record in train_dataset.records
    }

    val_classes = {
        record["class_index"]
        for record in val_dataset.records
    }

    assert len(train_classes) == 37
    assert len(val_classes) == 37