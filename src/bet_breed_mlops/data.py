from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(
    data_root: Path,
    batch_size: int = 32,
):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(
        data_root / "train",
        transform=transform,
    )

    val_dataset = datasets.ImageFolder(
        data_root / "val",
        transform=transform,
    )

    test_dataset = datasets.ImageFolder(
        data_root / "test",
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader