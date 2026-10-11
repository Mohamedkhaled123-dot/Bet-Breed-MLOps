from torch import Tensor, nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet18,
    resnet50,
)


NUM_CLASSES = 37


class PetBreedClassifier(nn.Module):
    """Image classifier for 37 Oxford-IIIT Pet breeds."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        self.backbone_name = backbone_name

        # --------------------------------------------------
        # ResNet-18
        # --------------------------------------------------

        if backbone_name == "resnet18":

            weights = (
                ResNet18_Weights.DEFAULT
                if pretrained
                else None
            )

            self.backbone = resnet18(
                weights=weights
            )

            if freeze_backbone:
                for parameter in self.backbone.parameters():
                    parameter.requires_grad = False

            in_features = self.backbone.fc.in_features

            self.backbone.fc = nn.Linear(
                in_features,
                num_classes,
            )

        # --------------------------------------------------
        # ResNet-50
        # --------------------------------------------------

        elif backbone_name == "resnet50":

            weights = (
                ResNet50_Weights.DEFAULT
                if pretrained
                else None
            )

            self.backbone = resnet50(weights=weights)

            if freeze_backbone:
                for parameter in self.backbone.parameters():
                    parameter.requires_grad = False

            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)

        # --------------------------------------------------
        # MobileNetV3-Small
        # --------------------------------------------------

        elif backbone_name == "mobilenet_v3_small":

            weights = (
                MobileNet_V3_Small_Weights.DEFAULT
                if pretrained
                else None
            )

            self.backbone = mobilenet_v3_small(
                weights=weights
            )

            if freeze_backbone:
                for parameter in self.backbone.parameters():
                    parameter.requires_grad = False

            in_features = self.backbone.classifier[-1].in_features

            self.backbone.classifier[-1] = nn.Linear(
                in_features,
                num_classes,
            )

        # --------------------------------------------------
        # EfficientNet-B0
        # --------------------------------------------------

        elif backbone_name == "efficientnet_b0":

            weights = (
                EfficientNet_B0_Weights.DEFAULT
                if pretrained
                else None
            )

            self.backbone = efficientnet_b0(
                weights=weights
            )

            if freeze_backbone:
                for parameter in self.backbone.parameters():
                    parameter.requires_grad = False

            in_features = self.backbone.classifier[-1].in_features

            self.backbone.classifier[-1] = nn.Linear(
                in_features,
                num_classes,
            )

        else:
            raise ValueError(
                f"Unsupported backbone: {backbone_name}. "
                "Choose from: resnet18, resnet50, "
                "mobilenet_v3_small, efficientnet_b0."
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)