"""U-shaped Neural Operator (U-NO) for 3D potential field prediction.

U-NO combines the multi-scale structure of U-Net with spectral convolutions
from FNO, enabling both local and global feature learning while maintaining
resolution independence.

Reference: Rahman et al., "U-NO: U-shaped Neural Operators" (TMLR 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .fno import SpectralConv3d


class UNOEncoderBlock(nn.Module):
    """Encoder block with spectral convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
    ):
        super().__init__()

        self.spectral_conv = SpectralConv3d(
            in_channels, out_channels, modes, modes, modes
        )
        self.local_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.GELU()
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning pooled output and skip connection."""
        # Spectral + local
        x1 = self.spectral_conv(x)
        x2 = self.local_conv(x)
        x = self.norm(x1 + x2)
        skip = self.activation(x)
        # Pool for next level
        x_down = self.pool(skip)
        return x_down, skip


class UNODecoderBlock(nn.Module):
    """Decoder block with spectral convolution and skip connection."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        modes: int,
    ):
        super().__init__()

        # Combined channels after concatenation
        combined_channels = in_channels + skip_channels

        self.spectral_conv = SpectralConv3d(
            combined_channels, out_channels, modes, modes, modes
        )
        self.local_conv = nn.Conv3d(combined_channels, out_channels, kernel_size=1)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection."""
        # Upsample to match skip connection
        x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        # Concatenate
        x = torch.cat([x, skip], dim=1)
        # Spectral + local
        x1 = self.spectral_conv(x)
        x2 = self.local_conv(x)
        x = self.norm(x1 + x2)
        x = self.activation(x)
        return x


class UNOBottleneck(nn.Module):
    """Bottleneck block at the coarsest scale."""

    def __init__(self, channels: int, modes: int):
        super().__init__()

        self.spectral_conv = SpectralConv3d(channels, channels, modes, modes, modes)
        self.local_conv = nn.Conv3d(channels, channels, kernel_size=1)
        self.norm = nn.InstanceNorm3d(channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.spectral_conv(x)
        x2 = self.local_conv(x)
        x = self.norm(x1 + x2)
        x = self.activation(x)
        return x


class UNO3D(nn.Module):
    """U-shaped Neural Operator for 3D problems.

    Combines multi-scale processing with spectral convolutions for
    both local detail and global coherence.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 1,
        base_width: int = 32,
        depth: int = 3,
        base_modes: int = 8,
        fc_dim: int = 128,
    ):
        """Initialize U-NO.

        Args:
            in_channels: Input feature channels
            out_channels: Output channels (1 for potential)
            base_width: Base channel width (doubled at each level)
            depth: Number of encoder/decoder levels
            base_modes: Fourier modes at finest level (halved at each level)
            fc_dim: FC decoder dimension
        """
        super().__init__()

        self.depth = depth

        # Channel progression
        self.channels = [base_width * (2 ** i) for i in range(depth + 1)]
        # Modes progression (decrease at coarser scales)
        self.modes = [max(2, base_modes // (2 ** i)) for i in range(depth + 1)]

        # Initial projection
        self.lift = nn.Conv3d(in_channels, base_width, kernel_size=1)

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        for i in range(depth):
            self.encoder_blocks.append(
                UNOEncoderBlock(
                    in_channels=self.channels[i],
                    out_channels=self.channels[i + 1],
                    modes=self.modes[i],
                )
            )

        # Bottleneck
        self.bottleneck = UNOBottleneck(
            channels=self.channels[-1],
            modes=self.modes[-1],
        )

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.decoder_blocks.append(
                UNODecoderBlock(
                    in_channels=self.channels[i + 1],
                    skip_channels=self.channels[i + 1],  # Skip from encoder
                    out_channels=self.channels[i],
                    modes=self.modes[i],
                )
            )

        # Projection
        self.proj1 = nn.Conv3d(base_width, fc_dim, kernel_size=1)
        self.proj2 = nn.Conv3d(fc_dim, out_channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-NO."""
        # Lift
        x = self.lift(x)

        # Encoder path
        skips = []
        for encoder in self.encoder_blocks:
            x, skip = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for i, decoder in enumerate(self.decoder_blocks):
            skip = skips[-(i + 1)]
            x = decoder(x, skip)

        # Project to output
        x = self.activation(self.proj1(x))
        x = self.proj2(x)

        return x


class UNOBackbone(nn.Module):
    """U-NO backbone wrapper."""

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 1,
        base_width: int = 32,
        depth: int = 3,
        base_modes: int = 8,
        fc_dim: int = 128,
    ):
        super().__init__()

        self.uno = UNO3D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_width=base_width,
            depth=depth,
            base_modes=base_modes,
            fc_dim=fc_dim,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.uno(features)
