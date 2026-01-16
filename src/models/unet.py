"""
3D U-Net backbone for potential field correction prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3d(nn.Module):
    """Double 3D convolution block with batch norm and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for predicting relative correction field.
    """
    
    def __init__(
        self,
        in_channels: int = 5,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            in_channels: Input channels (5: source, conductivity, X, Y, Z)
            base_channels: Base number of channels
            depth: Depth of U-Net (number of down/up sampling levels)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.depth = depth
        
        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        current_channels = in_channels
        
        for i in range(depth):
            out_channels = base_channels * (2 ** i)
            self.encoder.append(DoubleConv3d(current_channels, out_channels))
            current_channels = out_channels
        
        # Bottleneck
        bottleneck_channels = base_channels * (2 ** depth)
        self.bottleneck = DoubleConv3d(current_channels, bottleneck_channels)
        
        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        for i in range(depth - 1, -1, -1):
            in_channels_up = bottleneck_channels if i == depth - 1 else base_channels * (2 ** (i + 1))
            skip_channels = base_channels * (2 ** i)
            out_channels = base_channels * (2 ** i)
            
            self.upsamples.append(
                nn.ConvTranspose3d(in_channels_up, skip_channels, kernel_size=2, stride=2)
            )
            self.decoder.append(
                DoubleConv3d(in_channels_up, out_channels)
            )
        
        # Final output layer (predicts relative correction)
        self.final_conv = nn.Conv3d(base_channels, 1, kernel_size=1)
        self.dropout = nn.Dropout3d(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, 5, D, H, W)
        
        Returns:
            Relative correction field (B, 1, D, H, W)
        """
        # Encoder
        skip_connections = []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)
            x = F.max_pool3d(x, kernel_size=2, stride=2)
        
        # Bottleneck
        x = self.bottleneck(x)
        x = self.dropout(x)
        
        # Decoder
        for i, (upsample, decoder_block) in enumerate(zip(self.upsamples, self.decoder)):
            x = upsample(x)
            skip = skip_connections[-(i + 1)]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
            x = decoder_block(x)
        
        # Final output
        correction = self.final_conv(x)
        
        return correction
