import torch
import torch.nn as nn


class CMAFNet(nn.Module):

    def __init__(
        self,
        rgb_dim=512,
        thermal_dim=512,
        hidden_dim=256,
        num_tokens=16,
        num_heads=4,
        num_classes=27
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens

        # =================================================
        # Project RGB and Thermal features
        # =================================================

        self.rgb_projection = nn.Sequential(
            nn.Linear(rgb_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.thermal_projection = nn.Sequential(
            nn.Linear(thermal_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # =================================================
        # Convert projected feature into multiple tokens
        # =================================================

        self.rgb_token_projection = nn.Linear(
            hidden_dim,
            num_tokens * hidden_dim
        )

        self.thermal_token_projection = nn.Linear(
            hidden_dim,
            num_tokens * hidden_dim
        )

        # =================================================
        # Learnable positional embeddings
        # =================================================

        self.rgb_positional = nn.Parameter(
            torch.randn(1, num_tokens, hidden_dim)
        )

        self.thermal_positional = nn.Parameter(
            torch.randn(1, num_tokens, hidden_dim)
        )

        # =================================================
        # RGB -> Thermal cross attention
        # =================================================

        self.rgb_to_thermal = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # =================================================
        # Thermal -> RGB cross attention
        # =================================================

        self.thermal_to_rgb = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # =================================================
        # Token normalization
        # =================================================

        self.rgb_norm = nn.LayerNorm(
            hidden_dim
        )

        self.thermal_norm = nn.LayerNorm(
            hidden_dim
        )

        # =================================================
        # Token pooling
        # =================================================

        self.rgb_pool = nn.AdaptiveAvgPool1d(
            1
        )

        self.thermal_pool = nn.AdaptiveAvgPool1d(
            1
        )

        # =================================================
        # Fusion
        # =================================================

        self.fusion = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2)
        )

        # =================================================
        # Classifier
        # =================================================

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

        # =================================================
        # Project modalities
        # =================================================

        rgb_features = self.rgb_projection(
            rgb
        )

        thermal_features = self.thermal_projection(
            thermal
        )

        # =================================================
        # Generate RGB tokens
        # =================================================

        rgb_tokens = self.rgb_token_projection(
            rgb_features
        )

        rgb_tokens = rgb_tokens.view(
            -1,
            self.num_tokens,
            self.hidden_dim
        )

        rgb_tokens = (
            rgb_tokens
            +
            self.rgb_positional
        )

        # =================================================
        # Generate Thermal tokens
        # =================================================

        thermal_tokens = self.thermal_token_projection(
            thermal_features
        )

        thermal_tokens = thermal_tokens.view(
            -1,
            self.num_tokens,
            self.hidden_dim
        )

        thermal_tokens = (
            thermal_tokens
            +
            self.thermal_positional
        )

        # =================================================
        # RGB queries Thermal
        # =================================================

        rgb_attended, rgb_attention = (
            self.rgb_to_thermal(
                query=rgb_tokens,
                key=thermal_tokens,
                value=thermal_tokens,
                need_weights=True,
                average_attn_weights=False
            )
        )

        # =================================================
        # Thermal queries RGB
        # =================================================

        thermal_attended, thermal_attention = (
            self.thermal_to_rgb(
                query=thermal_tokens,
                key=rgb_tokens,
                value=rgb_tokens,
                need_weights=True,
                average_attn_weights=False
            )
        )

        # =================================================
        # Residual enhancement
        # =================================================

        enhanced_rgb = self.rgb_norm(
            rgb_tokens + rgb_attended
        )

        enhanced_thermal = self.thermal_norm(
            thermal_tokens + thermal_attended
        )

        # =================================================
        # Pool tokens
        # =================================================

        enhanced_rgb = enhanced_rgb.transpose(
            1,
            2
        )

        enhanced_thermal = enhanced_thermal.transpose(
            1,
            2
        )

        rgb_vector = self.rgb_pool(
            enhanced_rgb
        ).squeeze(
            -1
        )

        thermal_vector = self.thermal_pool(
            enhanced_thermal
        ).squeeze(
            -1
        )

        # =================================================
        # Fusion
        # =================================================

        fused = torch.cat(
            [
                rgb_vector,
                thermal_vector
            ],
            dim=1
        )

        fused = self.fusion(
            fused
        )

        # =================================================
        # Classification
        # =================================================

        output = self.classifier(
            fused
        )

        if return_attention:

            return (
                output,
                rgb_attention,
                thermal_attention
            )

        return output