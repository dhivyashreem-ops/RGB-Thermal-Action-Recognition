import torch
import torch.nn as nn


class FusionBaseline(nn.Module):

    def __init__(
        self,
        rgb_dim=512,
        thermal_dim=512,
        num_classes=27
    ):

        super().__init__()

        fusion_dim = rgb_dim + thermal_dim

        self.classifier = nn.Sequential(
            nn.Linear(
                fusion_dim,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                512,
                num_classes
            )
        )

    def forward(
        self,
        rgb_features,
        thermal_features
    ):

        fused = torch.cat(
            [
                rgb_features,
                thermal_features
            ],
            dim=1
        )

        output = self.classifier(
            fused
        )

        return output