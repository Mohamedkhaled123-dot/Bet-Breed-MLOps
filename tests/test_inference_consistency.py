from pathlib import Path

import torch
from PIL import Image

from bet_breed_mlops.data import eval_transform
from bet_breed_mlops.model import PetBreedClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT / "artifacts" / "checkpoints" / "resnet18.pt"
)

MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "metadata" / "manifest.json"
)


def test_eval_transform_and_model_are_deterministic():
    device = torch.device("cpu")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model = PetBreedClassifier(
        num_classes=37,
        backbone_name="resnet18",
        pretrained=False,
        freeze_backbone=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_path = None

    import json

    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    for record in manifest["images"]:
        if record["split"] == "val" and record["corruption"] is None:
            image_path = PROJECT_ROOT / record["path"]
            break

    assert image_path is not None

    image = Image.open(image_path).convert("RGB")

    input_tensor = eval_transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits_1 = model(input_tensor)
        logits_2 = model(input_tensor)

    max_diff = torch.max(
        torch.abs(logits_1 - logits_2)
    ).item()

    print(f"\nMax logits difference: {max_diff:.10f}")

    assert max_diff <= 1e-4