from pathlib import Path
import json

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = PROJECT_ROOT / "data" / "metadata" / "manifest.json"


# ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# Training: augmentation is allowed
train_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


# Validation: deterministic
eval_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

val_transform = eval_transform
class PetBreedDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        transform=None,
    ) -> None:

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.label_map = manifest["label_map"]
        self.records = [
            record
            for record in manifest["images"]
            if record["split"] == split
            and record["corruption"] is None
        ]

        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]

        image_path = PROJECT_ROOT / record["path"]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = record["class_index"]

        return image, label


def create_dataloaders(
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader]:

    train_dataset = PetBreedDataset(
        manifest_path=MANIFEST_PATH,
        split="train",
        transform=train_transform,
    )

    val_dataset = PetBreedDataset(
        manifest_path=MANIFEST_PATH,
        split="val",
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader