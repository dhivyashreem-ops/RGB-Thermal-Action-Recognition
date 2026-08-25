import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights


class ResNetFeatureExtractor(nn.Module):

    def __init__(self):

        super().__init__()

        weights = ResNet18_Weights.DEFAULT

        model = resnet18(
            weights=weights
        )

        self.features = nn.Sequential(
            *list(model.children())[:-1]
        )

        self.feature_dim = 512

        self.eval()

        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, x):

        with torch.no_grad():

            features = self.features(x)

        features = features.flatten(
            start_dim=1
        )

        return features