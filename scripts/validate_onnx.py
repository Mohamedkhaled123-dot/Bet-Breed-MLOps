from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from bet_breed_mlops.data import eval_transform
from bet_breed_mlops.model import PetBreedClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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


def load_pytorch_model() -> torch.nn.Module:
    model = PetBreedClassifier(
        num_classes=37,
        pretrained=False,
        freeze_backbone=True,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    return model


def load_image_tensor() -> torch.Tensor:
    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    tensor = eval_transform(image).unsqueeze(0)

    return tensor


def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"Image: {IMAGE_PATH.name}")
    print()

    # -------------------------
    # Prepare identical input
    # -------------------------
    input_tensor = load_image_tensor()

    # -------------------------
    # PyTorch inference
    # -------------------------
    model = load_pytorch_model()

    pytorch_input = input_tensor.to(DEVICE)

    with torch.inference_mode():
        pytorch_logits = model(
            pytorch_input
        )

    pytorch_logits = (
        pytorch_logits
        .detach()
        .cpu()
        .numpy()
    )

    # -------------------------
    # ONNX inference
    # -------------------------
    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    onnx_input_name = session.get_inputs()[0].name

    onnx_logits = session.run(
        None,
        {
            onnx_input_name: input_tensor.numpy(),
        },
    )[0]

    # -------------------------
    # Compare
    # -------------------------
    max_difference = np.max(
        np.abs(
            pytorch_logits - onnx_logits
        )
    )

    pytorch_class = int(
        np.argmax(pytorch_logits, axis=1)[0]
    )

    onnx_class = int(
        np.argmax(onnx_logits, axis=1)[0]
    )

    print("ONNX Validation")
    print("================")
    print(f"PyTorch class : {pytorch_class}")
    print(f"ONNX class    : {onnx_class}")
    print(f"Max logit diff: {max_difference:.8f}")

    if pytorch_class == onnx_class:
        print("Prediction match: PASS")
    else:
        print("Prediction match: FAIL")

    mean_difference = np.mean(
    np.abs(
        pytorch_logits - onnx_logits)
    )

    print(f"Mean logit diff: {mean_difference:.8f}")

    if max_difference < 0.01:
        print("Numerical difference: PASS")
    else:
        print("Numerical difference: FAIL")


if __name__ == "__main__":
    main()