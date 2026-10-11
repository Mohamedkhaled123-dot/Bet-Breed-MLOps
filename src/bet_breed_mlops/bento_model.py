import hashlib
from pathlib import Path

import bentoml
import onnx


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = PROJECT_ROOT / "artifacts" / "models" / "resnet18.onnx"
BENTO_MODEL_NAME = "pet-breed-classifier"


def register_bento_model() -> bentoml.Model:
    source_hash = hashlib.sha256(ONNX_PATH.read_bytes()).hexdigest()

    for model in bentoml.models.list():
        if (
            model.tag.name == BENTO_MODEL_NAME
            and model.info.metadata.get("source_sha256") == source_hash
        ):
            return model

    onnx_model = onnx.load(ONNX_PATH)
    return bentoml.onnx.save_model(
        BENTO_MODEL_NAME,
        onnx_model,
        signatures={
            "run": {
                "batchable": True,
                "batch_dim": 0,
            }
        },
        metadata={
            "source_sha256": source_hash,
            "source_path": "artifacts/models/resnet18.onnx",
            "input_name": onnx_model.graph.input[0].name,
            "num_classes": 37,
        },
    )
