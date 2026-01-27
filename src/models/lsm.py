"""Latent Spectral Model (LSM) for 3D potential field prediction.

LSM encodes the input to a low-dimensional latent space, performs spectral
processing in the latent space, and decodes back to the full resolution.
This approach is memory-efficient and can capture global patterns.

Reference: Wu et al., "Solving High-Dimensional PDEs with Latent Spectral Models" (ICML 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LatentEncoder(nn.Module):
    """Encoder that maps high-dimensional input to low-dimensional latent space."""

    def __init__(
        self,
        in_channels: int,
        latent_dim: int = 32,
        latent_resolution: Tuple[int, int, int] = (12, 6, 6),
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.latent_resolution = latent_resolution

        # Progressive downsampling with residual blocks
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden_dim),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(hidden_dim * 2),
            nn.GELU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(hidden_dim * 4),
            nn.GELU(),
        )

        # Adaptive pooling to fixed latent resolution
        self.adaptive_pool = nn.AdaptiveAvgPool3d(latent_resolution)

        # Project to latent dimension
        self.proj = nn.Conv3d(hidden_dim * 4, latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent space.

        Args:
            x: Input (B, C, D, H, W)

        Returns:
            Latent representation (B, latent_dim, Ld, Lh, Lw)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        x = self.proj(x)
        return x


class LatentDecoder(nn.Module):
    """Decoder that maps latent space back to full resolution."""

    def __init__(
        self,
        latent_dim: int = 32,
        out_channels: int = 1,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # Progressive upsampling
        self.proj = nn.Conv3d(latent_dim, hidden_dim * 4, kernel_size=1)

        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(hidden_dim * 2),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(hidden_dim),
            nn.GELU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(hidden_dim),
            nn.GELU(),
        )

        # Final projection
        self.final = nn.Conv3d(hidden_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
        """Decode from latent space.

        Args:
            x: Latent representation (B, latent_dim, Ld, Lh, Lw)
            target_shape: Target spatial dimensions (D, H, W)

        Returns:
            Decoded output (B, out_channels, D, H, W)
        """
        x = self.proj(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        # Interpolate to exact target size
        if x.shape[2:] != target_shape:
            x = F.interpolate(x, size=target_shape, mode='trilinear', align_corners=False)

        x = self.final(x)
        return x


class LatentSpectralConv3d(nn.Module):
    """Spectral convolution in latent space.

    Operates on the low-dimensional latent representation, making it
    more memory-efficient than full-resolution spectral convolutions.
    """

    def __init__(
        self,
        latent_dim: int,
        modes1: int = 6,
        modes2: int = 3,
        modes3: int = 3,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        scale = 1 / (latent_dim ** 2)

        # Weights for spectral convolution
        self.weights1 = nn.Parameter(
            scale * torch.randn(latent_dim, latent_dim, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(latent_dim, latent_dim, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights3 = nn.Parameter(
            scale * torch.randn(latent_dim, latent_dim, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights4 = nn.Parameter(
            scale * torch.randn(latent_dim, latent_dim, modes1, modes2, modes3, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral convolution in latent space."""
        B = x.shape[0]
        D, H, W = x.shape[2], x.shape[3], x.shape[4]

        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))

        out_ft = torch.zeros(
            B, self.latent_dim, D, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # Apply to quadrants
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, :self.modes1, :self.modes2, :self.modes3],
            self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3],
            self.weights2
        )
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3],
            self.weights3
        )
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3],
            self.weights4
        )

        x = torch.fft.irfftn(out_ft, s=(D, H, W), dim=(-3, -2, -1))
        return x


class LatentSpectralBlock(nn.Module):
    """Block for processing in latent space."""

    def __init__(
        self,
        latent_dim: int,
        modes1: int = 6,
        modes2: int = 3,
        modes3: int = 3,
    ):
        super().__init__()

        self.spectral = LatentSpectralConv3d(latent_dim, modes1, modes2, modes3)
        self.linear = nn.Conv3d(latent_dim, latent_dim, kernel_size=1)
        self.norm = nn.InstanceNorm3d(latent_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x1 = self.spectral(x)
        x2 = self.linear(x)
        x = self.norm(x1 + x2)
        x = self.activation(x + residual)
        return x


class LSM3D(nn.Module):
    """Latent Spectral Model for 3D potential field prediction.

    Processes in a compact latent space for efficiency while
    maintaining the ability to capture global patterns through
    spectral operations.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 1,
        latent_dim: int = 32,
        latent_resolution: Tuple[int, int, int] = (12, 6, 6),
        num_layers: int = 4,
        hidden_dim: int = 64,
        latent_modes: Tuple[int, int, int] = (6, 3, 3),
    ):
        """Initialize LSM.

        Args:
            in_channels: Input feature channels
            out_channels: Output channels
            latent_dim: Dimension of latent space
            latent_resolution: Spatial resolution of latent space
            num_layers: Number of spectral processing layers
            hidden_dim: Hidden dimension for encoder/decoder
            latent_modes: Fourier modes in latent space
        """
        super().__init__()

        self.latent_resolution = latent_resolution

        # Encoder
        self.encoder = LatentEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            latent_resolution=latent_resolution,
            hidden_dim=hidden_dim,
        )

        # Latent spectral processing
        self.latent_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.latent_layers.append(
                LatentSpectralBlock(
                    latent_dim=latent_dim,
                    modes1=latent_modes[0],
                    modes2=latent_modes[1],
                    modes3=latent_modes[2],
                )
            )

        # Decoder
        self.decoder = LatentDecoder(
            latent_dim=latent_dim,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSM."""
        target_shape = x.shape[2:]

        # Encode to latent space
        z = self.encoder(x)

        # Process in latent space
        for layer in self.latent_layers:
            z = layer(z)

        # Decode back to full resolution
        out = self.decoder(z, target_shape)

        return out


class LSMBackbone(nn.Module):
    """LSM backbone wrapper."""

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 1,
        latent_dim: int = 32,
        latent_resolution: Tuple[int, int, int] = (12, 6, 6),
        num_layers: int = 4,
        hidden_dim: int = 64,
        latent_modes: Tuple[int, int, int] = (6, 3, 3),
    ):
        super().__init__()

        self.lsm = LSM3D(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_dim=latent_dim,
            latent_resolution=latent_resolution,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            latent_modes=latent_modes,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.lsm(features)
