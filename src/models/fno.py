"""3D Fourier Neural Operator (FNO) backbone for potential field prediction.

This implementation uses rFFT (real FFT) for memory efficiency, exploiting
the conjugate symmetry of real-valued input signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SpectralConv3d(nn.Module):
    """3D Spectral Convolution layer using real FFT.

    Performs convolution in Fourier space by element-wise multiplication
    with learnable complex weights, truncated to specified mode counts.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
    ):
        """Initialize SpectralConv3d.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes1: Number of Fourier modes to keep in first dimension
            modes2: Number of Fourier modes to keep in second dimension
            modes3: Number of Fourier modes to keep in third dimension (rfft)
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)

        # Complex weights for different quadrants of the Fourier space
        # For rFFT, we only need half the modes in the last dimension
        # We need 4 sets of weights for the 4 quadrants (combinations of +/- modes in dims 1,2)
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights3 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights4 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )

    def compl_mul3d(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Complex multiplication in Fourier space.

        Args:
            input: (B, in_channels, x, y, z) complex tensor
            weights: (in_channels, out_channels, x, y, z) complex tensor

        Returns:
            Output tensor (B, out_channels, x, y, z)
        """
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral convolution.

        Args:
            x: Input tensor of shape (B, in_channels, D, H, W)

        Returns:
            Output tensor of shape (B, out_channels, D, H, W)
        """
        batch_size = x.shape[0]
        D, H, W = x.shape[2], x.shape[3], x.shape[4]

        # Compute real FFT (output has shape [..., W//2+1] due to conjugate symmetry)
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))

        # Allocate output tensor in Fourier space
        out_ft = torch.zeros(
            batch_size, self.out_channels,
            D, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # Apply spectral convolution to different quadrants
        # The modes are organized as:
        # dim1: [0, 1, ..., modes1-1] and [-modes1, ..., -1]
        # dim2: [0, 1, ..., modes2-1] and [-modes2, ..., -1]
        # dim3 (rfft): [0, 1, ..., modes3-1] only (due to conjugate symmetry)

        # Quadrant 1: positive modes1, positive modes2
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, :self.modes1, :self.modes2, :self.modes3],
            self.weights1
        )

        # Quadrant 2: negative modes1, positive modes2
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3],
            self.weights2
        )

        # Quadrant 3: positive modes1, negative modes2
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3],
            self.weights3
        )

        # Quadrant 4: negative modes1, negative modes2
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3],
            self.weights4
        )

        # Inverse real FFT
        x = torch.fft.irfftn(out_ft, s=(D, H, W), dim=(-3, -2, -1))

        return x


class FNOBlock(nn.Module):
    """Single FNO block: Spectral Conv + Linear bypass + Activation."""

    def __init__(
        self,
        width: int,
        modes1: int,
        modes2: int,
        modes3: int,
    ):
        """Initialize FNO block.

        Args:
            width: Number of channels
            modes1: Fourier modes in dim 1
            modes2: Fourier modes in dim 2
            modes3: Fourier modes in dim 3
        """
        super().__init__()

        self.spectral_conv = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.linear = nn.Conv3d(width, width, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor (B, width, D, H, W)

        Returns:
            Output tensor (B, width, D, H, W)
        """
        # Spectral path
        x1 = self.spectral_conv(x)
        # Linear bypass path
        x2 = self.linear(x)
        # Combine and activate
        x = self.activation(x1 + x2)
        return x


class FNO3D(nn.Module):
    """3D Fourier Neural Operator for potential field prediction.

    The FNO learns an operator mapping in function space, enabling
    zero-shot generalization to different resolutions at inference time.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 1,
        modes1: int = 8,
        modes2: int = 8,
        modes3: int = 8,
        width: int = 32,
        num_layers: int = 4,
        fc_dim: int = 128,
    ):
        """Initialize FNO3D.

        Args:
            in_channels: Number of input channels (from encoder)
            out_channels: Number of output channels (1 for potential)
            modes1: Fourier modes in dimension 1 (depth)
            modes2: Fourier modes in dimension 2 (height)
            modes3: Fourier modes in dimension 3 (width)
            width: Hidden channel width
            num_layers: Number of FNO layers
            fc_dim: Dimension of fully-connected decoder layers
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.num_layers = num_layers

        # Lifting layer: project inputs to FNO width
        self.lift = nn.Conv3d(in_channels, width, kernel_size=1)

        # FNO layers
        self.fno_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.fno_layers.append(
                FNOBlock(width, modes1, modes2, modes3)
            )

        # Projection layers (decoder)
        self.proj1 = nn.Conv3d(width, fc_dim, kernel_size=1)
        self.proj2 = nn.Conv3d(fc_dim, out_channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FNO.

        Args:
            x: Input features (B, in_channels, D, H, W)

        Returns:
            Predicted potential (B, out_channels, D, H, W)
        """
        # Lifting
        x = self.lift(x)

        # FNO layers
        for layer in self.fno_layers:
            x = layer(x)

        # Projection (decode)
        x = self.activation(self.proj1(x))
        x = self.proj2(x)

        return x


class FNOBackbone(nn.Module):
    """FNO-based backbone wrapper for potential field prediction."""

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 1,
        modes1: int = 8,
        modes2: int = 8,
        modes3: int = 8,
        width: int = 32,
        num_layers: int = 4,
        fc_dim: int = 128,
    ):
        """Initialize FNO backbone.

        Args:
            in_channels: Input channels from encoder
            out_channels: Output channels
            modes1: Fourier modes in dim 1
            modes2: Fourier modes in dim 2
            modes3: Fourier modes in dim 3
            width: Hidden width
            num_layers: Number of FNO layers
            fc_dim: FC decoder dimension
        """
        super().__init__()

        self.fno = FNO3D(
            in_channels=in_channels,
            out_channels=out_channels,
            modes1=modes1,
            modes2=modes2,
            modes3=modes3,
            width=width,
            num_layers=num_layers,
            fc_dim=fc_dim,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Encoded features (B, in_channels, D, H, W)

        Returns:
            Predicted potential (B, 1, D, H, W)
        """
        return self.fno(features)
