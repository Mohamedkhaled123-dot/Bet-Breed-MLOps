from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from bet_breed_mlops.data import eval_transform
from bet_breed_mlops.model import PetBreedClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "resnet18.pt"
)

ONNX_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "resnet18.onnx"
)

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "split"
    / "val"
    / "Abyssinian"
    / "Abyssinian_1.jpg"
)


def test_onnx_matches_pytorch():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load PyTorch model
    model = PetBreedClassifier(
        num_classes=37,
        pretrained=False,
        freeze_backbone=True,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    # Same preprocessing
    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    input_tensor = eval_transform(
        image
    ).unsqueeze(0)

    # PyTorch inference
    with torch.inference_mode():
        pytorch_logits = model(
            input_tensor.to(device)
        )

    pytorch_logits = (
        pytorch_logits
        .detach()
        .cpu()
        .numpy()
    )

    # ONNX inference
    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    input_name = session.get_inputs()[0].name

    onnx_logits = session.run(
        None,
        {
            input_name: input_tensor.numpy(),
        },
    )[0]

    # Compare predictions
    pytorch_class = int(
        np.argmax(
            pytorch_logits,
            axis=1,
        )[0]
    )

    onnx_class = int(
        np.argmax(
            onnx_logits,
            axis=1,
        )[0]
    )

    assert pytorch_class == onnx_class

    # Compare numerical difference
    max_difference = np.max(
        np.abs(
            pytorch_logits - onnx_logits
        )
    )

    assert max_difference < 0.01