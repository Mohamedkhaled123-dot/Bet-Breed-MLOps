from pathlib import Path

import mlflow
import torch

from bet_breed_mlops.model import PetBreedClassifier


mlflow.set_tracking_uri("http://127.0.0.1:5000")

EXPERIMENT_NAME = "bet-breed-backbone-comparison"

CHECKPOINT_DIR = Path("artifacts/checkpoints")

MODELS = {
    "resnet18": {
        "accuracy": 0.8967,
        "macro_f1": 0.8975,
        "parameters": 11_195_493,
        "checkpoint_size_mb": 42.78,
        "inference_ms": 1.75,
    },
    "mobilenet_v3_small": {
        "accuracy": 0.8438,
        "macro_f1": 0.8434,
        "parameters": 1_555_781,
        "checkpoint_size_mb": 6.07,
        "inference_ms": 3.46,
    },
    "efficientnet_b0": {
        "accuracy": 0.8859,
        "macro_f1": 0.8855,
        "parameters": 4_054_945,
        "checkpoint_size_mb": 15.76,
        "inference_ms": 5.28,
    },
}


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)

    for backbone_name, metrics in MODELS.items():
        checkpoint_path = CHECKPOINT_DIR / f"{backbone_name}.pt"

        with mlflow.start_run(run_name=backbone_name):
            mlflow.log_params(
                {
                    "backbone": backbone_name,
                    "pretrained": True,
                    "freeze_backbone": True,
                    "batch_size": 32,
                    "learning_rate": 1e-3,
                    "epochs": 5,
                    "optimizer": "Adam",
                    "image_size": 224,
                    "num_classes": 37,
                    "device": "cuda",
                }
            )

            mlflow.log_metrics(
                {
                    "val_accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "parameters": metrics["parameters"],
                    "checkpoint_size_mb": metrics["checkpoint_size_mb"],
                    "inference_ms": metrics["inference_ms"],
                }
            )

            mlflow.log_artifact(str(checkpoint_path))

            print(f"Logged: {backbone_name}")


if __name__ == "__main__":
    main()
    