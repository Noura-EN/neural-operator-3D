"""
Hybrid wrapper combining analytical solution with neural network correction.
"""

import torch
import torch.nn as nn
from typing import Optional

from .geometry import GeometryEncoder
from .unet import UNet3D
from .fno import FNO3D
from ..utils.analytical import analytical_potential


class HybridWrapper(nn.Module):
    """
    Hybrid model combining analytical solution with neural correction.
    
    Formula: Φ_total = Φ_analytical * (1 + Φ_correction)
    """
    
    def __init__(
        self,
        backbone: str = "fno",
        geometry_encoder: Optional[dict] = None,
        backbone_config: dict = None,
        analytical_config: dict = None
    ):
        """
        Args:
            backbone: Backbone type ("fno" or "unet")
            geometry_encoder: Geometry encoder config dict (None to disable)
            backbone_config: Configuration for backbone model
            analytical_config: Configuration for analytical solver
        """
        super().__init__()
        
        self.backbone_type = backbone
        self.analytical_config = analytical_config or {}
        
        # Geometry encoder (optional)
        if geometry_encoder and geometry_encoder.get('enabled', False):
            self.geometry_encoder = GeometryEncoder(
                in_channels=1,
                hidden_dim=geometry_encoder.get('hidden_dim', 64),
                num_layers=geometry_encoder.get('num_layers', 2)
            )
            geometry_out_channels = geometry_encoder.get('hidden_dim', 64)
        else:
            self.geometry_encoder = None
            geometry_out_channels = 0
        
        # Backbone network
        if backbone == "fno":
            fno_config = backbone_config or {}
            self.backbone = FNO3D(
                in_channels=5 + geometry_out_channels,  # 5 base channels + geometry features
                modes1=fno_config.get('modes1', 16),
                modes2=fno_config.get('modes2', 8),
                modes3=fno_config.get('modes3', 8),
                width=fno_config.get('width', 64),
                fc_dim=fno_config.get('fc_dim', 128),
                depth=fno_config.get('depth', 4)
            )
        elif backbone == "unet":
            unet_config = backbone_config or {}
            self.backbone = UNet3D(
                in_channels=5 + geometry_out_channels,
                base_channels=unet_config.get('base_channels', 32),
                depth=unet_config.get('depth', 4),
                dropout=unet_config.get('dropout', 0.1)
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
    
    def forward(
        self,
        input_tensor: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of hybrid model.
        
        Args:
            input_tensor: Input tensor (B, 5, D, H, W) where channels are
                         [source, conductivity, X, Y, Z]
            return_components: If True, return (total, analytical, correction)
        
        Returns:
            Total potential (B, 1, D, H, W) or tuple if return_components=True
        """
        B, C, D, H, W = input_tensor.shape
        device = input_tensor.device
        
        # Extract components
        source = input_tensor[:, 0:1, :, :, :]  # (B, 1, D, H, W)
        conductivity = input_tensor[:, 1:2, :, :, :]  # (B, 1, D, H, W)
        coords = input_tensor[:, 2:5, :, :, :]  # (B, 3, D, H, W)
        
        # Compute analytical potential
        phi_analytical = analytical_potential(
            coords=coords,
            source=source,
            conductivity=conductivity,
            current_I=self.analytical_config.get('current_I', 1.0),
            epsilon_factor=self.analytical_config.get('epsilon_factor', 0.1),
            coord_range=tuple(self.analytical_config.get('coord_range', [-1.0, 1.0]))
        )
        
        # Process through geometry encoder if enabled
        if self.geometry_encoder is not None:
            geometry_features = self.geometry_encoder(conductivity)  # (B, hidden_dim, D, H, W)
            # Concatenate geometry features to input
            backbone_input = torch.cat([input_tensor, geometry_features], dim=1)
        else:
            backbone_input = input_tensor
        
        # Get neural correction
        correction = self.backbone(backbone_input)  # (B, 1, D, H, W)
        
        # Combine: Φ_total = Φ_analytical * (1 + Φ_correction)
        phi_total = phi_analytical * (1.0 + correction)
        
        if return_components:
            return phi_total, phi_analytical, correction
        else:
            return phi_total
