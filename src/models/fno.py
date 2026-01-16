"""
3D Fourier Neural Operator (FNO) for potential field correction prediction.
Supports zero-shot super-resolution via resolution-independent coordinate grids.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralConv3d(nn.Module):
    """
    3D Spectral Convolution layer for FNO.
    Performs convolution in Fourier space for resolution-independent operations.
    """
    
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, modes3: int):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            modes1, modes2, modes3: Number of Fourier modes to keep in each dimension
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        
        self.scale = (1 / (in_channels * out_channels))
        
        # Learnable weights for low-frequency modes
        # Each weight tensor has shape (in_channels, out_channels, modes, modes, modes, 2)
        # where the last dimension is [real, imag]
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, 2)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, 2)
        )
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, 2)
        )
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, 2)
        )
        self.weights5 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, 2)
        )
        self.weights6 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, 2)
        )
    
    def compl_mul3d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Complex multiplication in 3D.
        
        Args:
            input: Complex tensor (B, in_channels, D, H, W//2+1)
            weights: Complex weights (in_channels, out_channels, modes1, modes2, modes3)
        
        Returns:
            Complex tensor (B, out_channels, D, H, W//2+1)
        """
        # Extract the relevant slice dimensions from input
        B, in_ch, D_slice, H_slice, W_slice = input.shape
        out_ch = weights.shape[1]
        
        # Reshape for einsum: (B, in_ch, D, H, W) x (in_ch, out_ch, D, H, W) -> (B, out_ch, D, H, W)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, in_channels, D, H, W)
        
        Returns:
            Output tensor (B, out_channels, D, H, W)
        """
        batchsize = x.shape[0]
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1], norm='ortho')
        
        # Get dimensions
        D, H, W_half = x_ft.shape[-3:]
        W = (W_half - 1) * 2  # Original W dimension
        
        # Initialize output in Fourier space
        out_ft = torch.zeros(
            batchsize, self.out_channels, D, H, W_half,
            dtype=torch.cfloat, device=x.device
        )
        
        # Convert weights to complex
        weights1_c = torch.complex(self.weights1[..., 0], self.weights1[..., 1])
        weights2_c = torch.complex(self.weights2[..., 0], self.weights2[..., 1])
        weights3_c = torch.complex(self.weights3[..., 0], self.weights3[..., 1])
        weights4_c = torch.complex(self.weights4[..., 0], self.weights4[..., 1])
        weights5_c = torch.complex(self.weights5[..., 0], self.weights5[..., 1])
        weights6_c = torch.complex(self.weights6[..., 0], self.weights6[..., 1])
        
        # Multiply relevant Fourier modes
        # Low frequencies: [0:modes1, 0:modes2, 0:modes3]
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(
                x_ft[:, :, :self.modes1, :self.modes2, :self.modes3],
                weights1_c
            )
        
        # High frequencies in D: [-modes1:, 0:modes2, 0:modes3]
        if D > self.modes1:
            out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
                self.compl_mul3d(
                    x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3],
                    weights2_c
                )
        
        # High frequencies in H: [0:modes1, -modes2:, 0:modes3]
        if H > self.modes2:
            out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
                self.compl_mul3d(
                    x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3],
                    weights3_c
                )
        
        # High frequencies in D and H: [-modes1:, -modes2:, 0:modes3]
        if D > self.modes1 and H > self.modes2:
            out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
                self.compl_mul3d(
                    x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3],
                    weights4_c
                )
        
        # High frequencies in W: [0:modes1, 0:modes2, -modes3:]
        if W_half > self.modes3:
            out_ft[:, :, :self.modes1, :self.modes2, -self.modes3:] = \
                self.compl_mul3d(
                    x_ft[:, :, :self.modes1, :self.modes2, -self.modes3:],
                    weights5_c
                )
        
        # High frequencies in all dimensions
        if D > self.modes1 and H > self.modes2 and W_half > self.modes3:
            out_ft[:, :, -self.modes1:, -self.modes2:, -self.modes3:] = \
                self.compl_mul3d(
                    x_ft[:, :, -self.modes1:, -self.modes2:, -self.modes3:],
                    weights6_c
                )
        
        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(D, H, W), dim=[-3, -2, -1], norm='ortho')
        
        return x


class FNO3D(nn.Module):
    """
    3D Fourier Neural Operator for predicting relative correction field.
    Supports zero-shot super-resolution via resolution-independent coordinate grids.
    """
    
    def __init__(
        self,
        in_channels: int = 5,
        modes1: int = 16,
        modes2: int = 8,
        modes3: int = 8,
        width: int = 64,
        fc_dim: int = 128,
        depth: int = 4
    ):
        """
        Args:
            in_channels: Input channels (5: source, conductivity, X, Y, Z)
            modes1, modes2, modes3: Number of Fourier modes in each dimension
            width: Width of the network
            fc_dim: Dimension of fully connected layers
            depth: Number of Fourier layers
        """
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.depth = depth
        
        # Lift input to higher dimension
        self.fc0 = nn.Linear(in_channels, self.width)
        
        # Fourier layers
        self.conv_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        
        for i in range(self.depth):
            self.conv_layers.append(
                SpectralConv3d(self.width, self.width, modes1, modes2, modes3)
            )
            self.w_layers.append(
                nn.Conv3d(self.width, self.width, kernel_size=1)
            )
        
        # Project to output dimension
        self.fc1 = nn.Linear(self.width, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with resolution-independent coordinate grids.
        
        Args:
            x: Input tensor (B, 5, D, H, W) where channels are [source, conductivity, X, Y, Z]
        
        Returns:
            Relative correction field (B, 1, D, H, W)
        """
        batch_size = x.shape[0]
        grid_size = x.shape[2:]  # (D, H, W)
        
        # Reshape input to (B, D, H, W, 5)
        x = x.permute(0, 2, 3, 4, 1)
        
        # Lift to higher dimension: (B, D, H, W, width)
        x = self.fc0(x)
        
        # Reshape to (B, width, D, H, W)
        x = x.permute(0, 4, 1, 2, 3)
        
        # Apply Fourier layers
        for i in range(self.depth):
            x1 = self.conv_layers[i](x)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            if i < self.depth - 1:
                x = F.gelu(x)
        
        # Reshape to (B, D, H, W, width)
        x = x.permute(0, 2, 3, 4, 1)
        
        # Project to output dimension
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        # Reshape back to (B, 1, D, H, W)
        x = x.permute(0, 4, 1, 2, 3)
        
        return x
