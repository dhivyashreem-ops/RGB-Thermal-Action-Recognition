import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class RGBBaseline(nn.Module):

    def __init__(
        self,
        num_classes=27
    ):

        super().__init__()

        self.backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        feature_dim = (
            self.backbone.fc.in_features
        )

        self.backbone.fc = nn.Identity()

        self.classifier = nn.Linear(
            feature_dim,
            num_classes
        )

    def forward(self, x):

        # x:
        # [B, T, C, H, W]

        batch_size, time_steps, channels, height, width = x.shape

        x = x.reshape(
            batch_size * time_steps,
            channels,
            height,
            width
        )

        features = self.backbone(x)

        features = features.reshape(
            batch_size,
            time_steps,
            -1
        )

        # Temporal average
        features = features.mean(
            dim=1
        )

        output = self.classifier(
            features
        )

        return output