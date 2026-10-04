from pathlib import Path

import mlflow
import mlflow.pytorch
import torch

from bet_breed_mlops.model import PetBreedClassifier


# --------------------------------------------------
# Configuration
# --------------------------------------------------

mlflow.set_tracking_uri("http://127.0.0.1:5000")

EXPERIMENT_NAME = "bet-breed-production-candidate"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "resnet18.pt"
)

CALIBRATION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "calibration"
    / "temperature.pt"
)

MODEL_NAME = "PetBreedClassifier"


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:

    mlflow.set_experiment(EXPERIMENT_NAME)

    # ----------------------------------------------
    # Load checkpoint
    # ----------------------------------------------

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

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # ----------------------------------------------
    # Load calibration
    # ----------------------------------------------

    calibration = torch.load(
        CALIBRATION_PATH,
        map_location="cpu",
        weights_only=False,
    )

    temperature = float(
        calibration["temperature"]
    )

    ece_before = float(
        calibration["ece_before"]
    )

    ece_after = float(
        calibration["ece_after"]
    )

    # ----------------------------------------------
    # Start MLflow run
    # ----------------------------------------------

    with mlflow.start_run(
        run_name="resnet18-production-candidate"
    ):

        # ------------------------------------------
        # Parameters
        # ------------------------------------------

        mlflow.log_params(
            {
                "backbone": "resnet18",
                "pretrained": True,
                "freeze_backbone": True,
                "image_size": 224,
                "num_classes": 37,
                "temperature": temperature,
                "confidence_threshold": 0.85,
            }
        )

        # ------------------------------------------
        # Metrics
        # ------------------------------------------

        mlflow.log_metrics(
            {
                "val_accuracy": 0.8967,
                "macro_f1": 0.8975,
                "ece_before": ece_before,
                "ece_after": ece_after,
                "coverage": 0.6522,
                "selective_accuracy": 0.9896,
            }
        )

        # ------------------------------------------
        # Checkpoint artifact
        # ------------------------------------------

        mlflow.log_artifact(
            str(CHECKPOINT_PATH),
            artifact_path="checkpoint",
        )

        # ------------------------------------------
        # Calibration artifact
        # ------------------------------------------

        mlflow.log_artifact(
            str(CALIBRATION_PATH),
            artifact_path="calibration",
        )

        # ------------------------------------------
        # Log MLflow model
        # ------------------------------------------
        example_input = torch.randn(1,3,224,224,)


        mlflow.pytorch.log_model(
            model,
            name="model",
            input_example=example_input,
            serialization_format="pickle",

        )

        run_id = mlflow.active_run().info.run_id

        print()
        print("Production candidate logged successfully.")
        print()
        print(f"Run ID: {run_id}")
        print(f"Model: {MODEL_NAME}")
        print("Backbone: resnet18")
        print(f"Temperature: {temperature:.4f}")
        print(f"ECE before: {ece_before:.4f}")
        print(f"ECE after: {ece_after:.4f}")
        print("Threshold: 0.85")
        print("Coverage: 0.6522")
        print("Selective accuracy: 0.9896")


if __name__ == "__main__":
    main()