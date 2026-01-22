"""Geometry Encoder for processing conductivity fields."""

import torch
import torch.nn as nn
from typing import Optional


class GeometryEncoder(nn.Module):
    """3D convolutional encoder for processing conductivity tensor fields.

    Encodes the spatially-varying conductivity tensor into a high-dimensional
    feature volume that can be used by downstream backbones (UNet, FNO).
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_dim: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 2,
    ):
        """Initialize GeometryEncoder.

        Args:
            in_channels: Number of input channels (6 for symmetric 3x3 tensor)
            hidden_dim: Hidden dimension for intermediate layers
            out_channels: Output channels (defaults to hidden_dim)
            num_layers: Number of convolutional layers
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels if out_channels is not None else hidden_dim
        self.num_layers = num_layers

        # Build encoder layers
        layers = []

        # First layer: in_channels -> hidden_dim
        layers.append(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False)
        )
        layers.append(nn.BatchNorm3d(hidden_dim))
        layers.append(nn.GELU())

        # Intermediate layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            layers.append(
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
            )
            layers.append(nn.BatchNorm3d(hidden_dim))
            layers.append(nn.GELU())

        # Final layer: hidden_dim -> out_channels
        if num_layers > 1:
            layers.append(
                nn.Conv3d(hidden_dim, self.out_channels, kernel_size=3, padding=1, bias=False)
            )
            layers.append(nn.BatchNorm3d(self.out_channels))
            layers.append(nn.GELU())

        self.encoder = nn.Sequential(*layers)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """Encode conductivity tensor field.

        Args:
            sigma: Conductivity tensor of shape (B, 6, D, H, W)

        Returns:
            Encoded features of shape (B, out_channels, D, H, W)
        """
        return self.encoder(sigma)


class SpacingConditioner(nn.Module):
    """Conditioning module to inject voxel spacing information into features.

    This module creates a spatial modulation based on the physical voxel spacing,
    which is critical for resolution-independent operator learning.

    Uses additive conditioning: features + mlp(spacing)
    This was found to be the best performing mode in ablation studies.
    """

    def __init__(
        self,
        spacing_dim: int = 3,
        feature_dim: int = 64,
        hidden_dim: int = 32,
    ):
        """Initialize SpacingConditioner.

        Args:
            spacing_dim: Dimension of spacing vector (3 for dx, dy, dz)
            feature_dim: Dimension of features to condition
            hidden_dim: Hidden dimension for the MLP
        """
        super().__init__()
        self.feature_dim = feature_dim

        # Additive conditioning: features + mlp(spacing)
        self.mlp = nn.Sequential(
            nn.Linear(spacing_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
            # No activation - can be positive or negative
        )

    def forward(
        self,
        features: torch.Tensor,
        spacing: torch.Tensor,
    ) -> torch.Tensor:
        """Apply spacing-based conditioning to features.

        Args:
            features: Feature tensor of shape (B, C, D, H, W)
            spacing: Spacing vector of shape (B, 3)

        Returns:
            Conditioned features of shape (B, C, D, H, W)
        """
        bias = self.mlp(spacing)  # (B, C)
        bias = bias.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return features + bias


class CombinedEncoder(nn.Module):
    """Combined encoder that processes geometry and other inputs together.

    This module:
    1. Encodes the conductivity tensor via GeometryEncoder
    2. Concatenates with source field, coordinates
    3. Applies additive spacing conditioning (found to be optimal in ablations)
    """

    def __init__(
        self,
        sigma_channels: int = 6,
        source_channels: int = 1,
        coord_channels: int = 3,
        geometry_hidden_dim: int = 64,
        geometry_num_layers: int = 2,
        out_channels: int = 64,
    ):
        """Initialize CombinedEncoder.

        Args:
            sigma_channels: Number of conductivity channels
            source_channels: Number of source field channels
            coord_channels: Number of coordinate channels (3 for X, Y, Z)
            geometry_hidden_dim: Hidden dimension for geometry encoder
            geometry_num_layers: Number of layers in geometry encoder
            out_channels: Output channels
        """
        super().__init__()

        self.geometry_encoder = GeometryEncoder(
            in_channels=sigma_channels,
            hidden_dim=geometry_hidden_dim,
            out_channels=geometry_hidden_dim,
            num_layers=geometry_num_layers,
        )

        # Input to fusion: geometry features + source + coords
        fusion_in_channels = geometry_hidden_dim + source_channels + coord_channels

        self.fusion = nn.Sequential(
            nn.Conv3d(fusion_in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        )

        # Additive spacing conditioning (best performing mode)
        self.spacing_conditioner = SpacingConditioner(
            spacing_dim=3,
            feature_dim=out_channels,
        )

        self.out_channels = out_channels

    def forward(
        self,
        sigma: torch.Tensor,
        source: torch.Tensor,
        coords: torch.Tensor,
        spacing: torch.Tensor,
    ) -> torch.Tensor:
        """Encode all inputs into a combined feature volume.

        Args:
            sigma: Conductivity tensor (B, 6, D, H, W)
            source: Source field (B, 1, D, H, W)
            coords: Coordinates (B, 3, D, H, W)
            spacing: Voxel spacing (B, 3)

        Returns:
            Encoded features (B, out_channels, D, H, W)
        """
        # Encode geometry
        geom_features = self.geometry_encoder(sigma)  # (B, hidden_dim, D, H, W)

        # Concatenate all inputs
        combined = torch.cat([geom_features, source, coords], dim=1)

        # Fuse features
        features = self.fusion(combined)

        # Apply additive spacing conditioning
        features = self.spacing_conditioner(features, spacing)

        return features
