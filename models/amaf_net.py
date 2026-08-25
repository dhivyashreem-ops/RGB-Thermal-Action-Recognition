import torch
import torch.nn as nn


class AMAFNet(nn.Module):

    def __init__(
        self,
        rgb_dim=512,
        thermal_dim=512,
        hidden_dim=256,
        num_classes=27
    ):

        super().__init__()

        # -------------------------------------------------
        # RGB projection
        # -------------------------------------------------

        self.rgb_projection = nn.Sequential(

            nn.Linear(
                rgb_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.LayerNorm(
                hidden_dim
            )
        )


        # -------------------------------------------------
        # Thermal projection
        # -------------------------------------------------

        self.thermal_projection = nn.Sequential(

            nn.Linear(
                thermal_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.LayerNorm(
                hidden_dim
            )
        )


        # -------------------------------------------------
        # Adaptive modality attention
        # -------------------------------------------------

        self.attention = nn.Sequential(

            nn.Linear(
                hidden_dim * 2,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                2
            )
        )


        # -------------------------------------------------
        # Classifier
        # -------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_dim,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                num_classes
            )
        )


    def forward(
        self,
        rgb,
        thermal,
        return_attention=False
    ):

        # Project both modalities

        rgb_features = self.rgb_projection(
            rgb
        )

        thermal_features = self.thermal_projection(
            thermal
        )


        # Generate modality attention

        combined = torch.cat(
            [
                rgb_features,
                thermal_features
            ],
            dim=1
        )


        attention_logits = self.attention(
            combined
        )


        attention_weights = torch.softmax(
            attention_logits,
            dim=1
        )


        # Adaptive fusion

        rgb_weight = attention_weights[
            :, 0:1
        ]

        thermal_weight = attention_weights[
            :, 1:2
        ]


        fused = (
            rgb_weight * rgb_features
            +
            thermal_weight * thermal_features
        )


        # Classification

        output = self.classifier(
            fused
        )


        if return_attention:

            return (
                output,
                attention_weights
            )


        return output