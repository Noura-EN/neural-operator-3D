"""3D UNet backbone for potential field prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ConvBlock(nn.Module):
    """Basic 3D convolutional block with BatchNorm and GELU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.activation(self.bn2(self.conv2(x)))
        return x


class DownBlock(nn.Module):
    """Downsampling block: MaxPool + ConvBlock."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block: Upsample + Concatenate skip + ConvBlock."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        # Upsample to match skip connection size
        x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNet3D(nn.Module):
    """3D UNet backbone for potential field prediction.

    Standard encoder-decoder architecture with skip connections.
    Suitable as a baseline for validating the pipeline.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize 3D UNet.

        Args:
            in_channels: Number of input channels (from CombinedEncoder)
            out_channels: Number of output channels (1 for potential field)
            base_channels: Base number of channels (doubles at each level)
            depth: Number of downsampling levels
            dropout: Dropout probability
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        # Calculate channel sizes at each level
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]

        # Initial convolution
        self.init_conv = ConvBlock(in_channels, channels[0], dropout=dropout)

        # Encoder (downsampling path)
        self.encoders = nn.ModuleList()
        for i in range(depth):
            self.encoders.append(
                DownBlock(channels[i], channels[i + 1], dropout=dropout)
            )

        # Decoder (upsampling path)
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.decoders.append(
                UpBlock(channels[i + 1], channels[i], channels[i], dropout=dropout)
            )

        # Output convolution
        self.out_conv = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through UNet.

        Args:
            x: Input features of shape (B, in_channels, D, H, W)

        Returns:
            Output potential field of shape (B, out_channels, D, H, W)
        """
        # Initial convolution
        x = self.init_conv(x)

        # Encoder path - save skip connections
        skips = [x]
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        # Remove the last skip (it's the bottleneck)
        skips = skips[:-1]

        # Decoder path - use skip connections in reverse order
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        # Output
        x = self.out_conv(x)

        return x


class UNetBackbone(nn.Module):
    """Full UNet-based model for potential field prediction.

    This wraps the core UNet with input encoding.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize UNet backbone.

        Args:
            in_channels: Number of input channels from encoder
            out_channels: Number of output channels
            base_channels: Base channel count
            depth: UNet depth
            dropout: Dropout probability
        """
        super().__init__()

        self.unet = UNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            depth=depth,
            dropout=dropout,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Encoded features (B, in_channels, D, H, W)

        Returns:
            Predicted potential (B, 1, D, H, W)
        """
        return self.unet(features)
