"""
Geometry encoder for processing structural features of conductivity maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryEncoder(nn.Module):
    """
    Encoder to process conductivity volume into latent feature maps.
    Extracts structural/geometric features before passing to backbone.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        kernel_size: int = 3
    ):
        """
        Args:
            in_channels: Input channels (typically 1 for conductivity)
            hidden_dim: Hidden dimension for feature maps
            num_layers: Number of convolutional layers
            kernel_size: Convolution kernel size
        """
        super().__init__()
        
        layers = []
        current_channels = in_channels
        
        for i in range(num_layers):
            layers.extend([
                nn.Conv3d(
                    current_channels,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            current_channels = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, conductivity: torch.Tensor) -> torch.Tensor:
        """
        Args:
            conductivity: Conductivity tensor (B, 1, D, H, W)
        
        Returns:
            Encoded features (B, hidden_dim, D, H, W)
        """
        return self.encoder(conductivity)
