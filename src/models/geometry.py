"""Geometry Encoder for processing conductivity fields."""

import torch
import torch.nn as nn
from typing import Optional


class ResidualBlock3d(nn.Module):
    """3D Residual block with pre-activation (BatchNorm -> GELU -> Conv)."""

    def __init__(self, channels: int):
        super().__init__()
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.activation(self.bn1(x))
        out = self.conv1(out)
        out = self.activation(self.bn2(out))
        out = self.conv2(out)
        return out + residual


class GeometryEncoder(nn.Module):
    """3D convolutional encoder for processing conductivity tensor fields.

    Encodes the spatially-varying conductivity tensor into a high-dimensional
    feature volume that can be used by downstream backbones (UNet, FNO).

    Supports both plain sequential and residual architectures.
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_dim: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 2,
        use_residual: bool = False,
    ):
        """Initialize GeometryEncoder.

        Args:
            in_channels: Number of input channels (6 for symmetric 3x3 tensor)
            hidden_dim: Hidden dimension for intermediate layers
            out_channels: Output channels (defaults to hidden_dim)
            num_layers: Number of convolutional layers (or residual blocks if use_residual=True)
            use_residual: If True, use residual blocks for deeper networks
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels if out_channels is not None else hidden_dim
        self.num_layers = num_layers
        self.use_residual = use_residual

        if use_residual:
            # ResNet-style architecture for deeper networks
            # Stem: project input to hidden_dim
            self.stem = nn.Sequential(
                nn.Conv3d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.GELU(),
            )

            # Residual blocks
            self.res_blocks = nn.ModuleList([
                ResidualBlock3d(hidden_dim) for _ in range(num_layers)
            ])

            # Output projection if needed
            if self.out_channels != hidden_dim:
                self.out_proj = nn.Sequential(
                    nn.BatchNorm3d(hidden_dim),
                    nn.GELU(),
                    nn.Conv3d(hidden_dim, self.out_channels, kernel_size=1, bias=False),
                )
            else:
                self.out_proj = nn.Sequential(
                    nn.BatchNorm3d(hidden_dim),
                    nn.GELU(),
                )
        else:
            # Original sequential architecture
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
        if self.use_residual:
            x = self.stem(sigma)
            for block in self.res_blocks:
                x = block(x)
            return self.out_proj(x)
        else:
            return self.encoder(sigma)


class SpacingConditioner(nn.Module):
    """Conditioning module to inject voxel spacing information into features.

    This module creates a spatial modulation based on the physical voxel spacing,
    which is critical for resolution-independent operator learning.

    Uses additive conditioning: features + mlp(transform(spacing))
    Supports different spacing transformations for better generalization.
    """

    def __init__(
        self,
        spacing_dim: int = 3,
        feature_dim: int = 64,
        hidden_dim: int = 32,
        spacing_transform: str = "none",
        reference_spacing: float = 2.0,
    ):
        """Initialize SpacingConditioner.

        Args:
            spacing_dim: Dimension of spacing vector (3 for dx, dy, dz)
            feature_dim: Dimension of features to condition
            hidden_dim: Hidden dimension for the MLP
            spacing_transform: Transform to apply to spacing before MLP
                - "none": Use raw spacing values (default)
                - "log": Apply log transform (better for extrapolation)
                - "normalized": Divide by reference_spacing
            reference_spacing: Reference spacing for normalization (default: 2.0mm)
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.spacing_transform = spacing_transform
        self.reference_spacing = reference_spacing

        # Additive conditioning: features + mlp(spacing)
        self.mlp = nn.Sequential(
            nn.Linear(spacing_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
            # No activation - can be positive or negative
        )

    def _transform_spacing(self, spacing: torch.Tensor) -> torch.Tensor:
        """Apply transformation to spacing values."""
        if self.spacing_transform == "log":
            # Log transform: compresses range, better for extrapolation
            # Add small epsilon to avoid log(0)
            return torch.log(spacing + 1e-6)
        elif self.spacing_transform == "normalized":
            # Normalize to reference spacing
            return spacing / self.reference_spacing
        else:
            # No transform
            return spacing

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
        # Apply transform to spacing
        transformed_spacing = self._transform_spacing(spacing)

        bias = self.mlp(transformed_spacing)  # (B, C)
        bias = bias.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return features + bias


class CombinedEncoder(nn.Module):
    """Combined encoder that processes geometry and other inputs together.

    This module:
    1. Encodes the conductivity tensor via GeometryEncoder
    2. Concatenates with source field, coordinates, and optional distance field
    3. Optionally applies additive spacing conditioning (found to be optimal in ablations)
    """

    def __init__(
        self,
        sigma_channels: int = 6,
        source_channels: int = 1,
        coord_channels: int = 3,
        distance_channels: int = 0,
        geometry_hidden_dim: int = 64,
        geometry_num_layers: int = 2,
        geometry_use_residual: bool = False,
        out_channels: int = 64,
        use_spacing_conditioning: bool = True,
        spacing_transform: str = "none",
        reference_spacing: float = 2.0,
    ):
        """Initialize CombinedEncoder.

        Args:
            sigma_channels: Number of conductivity channels
            source_channels: Number of source field channels
            coord_channels: Number of coordinate channels (3 for X, Y, Z)
            distance_channels: Number of distance field channels (0 or 1)
            geometry_hidden_dim: Hidden dimension for geometry encoder
            geometry_num_layers: Number of layers in geometry encoder
            geometry_use_residual: Whether to use residual blocks in geometry encoder
            out_channels: Output channels
            use_spacing_conditioning: Whether to apply spacing-based conditioning
            spacing_transform: Transform for spacing values ("none", "log", "normalized")
            reference_spacing: Reference spacing for normalization (default: 2.0mm)
        """
        super().__init__()

        self.use_spacing_conditioning = use_spacing_conditioning
        self.distance_channels = distance_channels

        self.geometry_encoder = GeometryEncoder(
            in_channels=sigma_channels,
            hidden_dim=geometry_hidden_dim,
            out_channels=geometry_hidden_dim,
            num_layers=geometry_num_layers,
            use_residual=geometry_use_residual,
        )

        # Input to fusion: geometry features + source + coords + distance
        fusion_in_channels = geometry_hidden_dim + source_channels + coord_channels + distance_channels

        self.fusion = nn.Sequential(
            nn.Conv3d(fusion_in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        )

        # Additive spacing conditioning (best performing mode)
        # Only create if enabled
        if use_spacing_conditioning:
            self.spacing_conditioner = SpacingConditioner(
                spacing_dim=3,
                feature_dim=out_channels,
                spacing_transform=spacing_transform,
                reference_spacing=reference_spacing,
            )
        else:
            self.spacing_conditioner = None

        self.out_channels = out_channels

    def forward(
        self,
        sigma: torch.Tensor,
        source: torch.Tensor,
        coords: torch.Tensor,
        spacing: torch.Tensor,
        distance_field: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode all inputs into a combined feature volume.

        Args:
            sigma: Conductivity tensor (B, 6, D, H, W)
            source: Source field (B, 1, D, H, W)
            coords: Coordinates (B, 3, D, H, W)
            spacing: Voxel spacing (B, 3)
            distance_field: Optional signed distance to boundary (B, 1, D, H, W)

        Returns:
            Encoded features (B, out_channels, D, H, W)
        """
        # Encode geometry
        geom_features = self.geometry_encoder(sigma)  # (B, hidden_dim, D, H, W)

        # Concatenate all inputs
        inputs_to_concat = [geom_features, source, coords]
        if distance_field is not None and self.distance_channels > 0:
            inputs_to_concat.append(distance_field)
        combined = torch.cat(inputs_to_concat, dim=1)

        # Fuse features
        features = self.fusion(combined)

        # Apply additive spacing conditioning if enabled
        if self.use_spacing_conditioning and self.spacing_conditioner is not None:
            features = self.spacing_conditioner(features, spacing)

        return features
