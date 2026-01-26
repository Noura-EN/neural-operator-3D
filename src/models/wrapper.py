"""Model wrapper that combines encoder and backbone for potential field prediction."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union

from .geometry import CombinedEncoder
from .unet import UNetBackbone
from .fno import FNOBackbone


class PotentialFieldModel(nn.Module):
    """Unified model for potential field prediction.

    Combines:
    1. CombinedEncoder: Processes conductivity, source, and coordinates
       with additive spacing conditioning
    2. Backbone: Either UNet or FNO for predicting the potential field
    """

    def __init__(
        self,
        backbone_type: str = "fno",
        sigma_channels: int = 6,
        source_channels: int = 1,
        coord_channels: int = 3,
        out_channels: int = 1,
        geometry_hidden_dim: int = 64,
        geometry_num_layers: int = 2,
        unet_config: Optional[Dict] = None,
        fno_config: Optional[Dict] = None,
        add_analytical_solution: bool = False,
        use_spacing_conditioning: bool = True,
        spacing_transform: str = "none",
        reference_spacing: float = 2.0,
        residual_learning: bool = False,
    ):
        """Initialize the potential field model.

        Args:
            backbone_type: Type of backbone ("unet" or "fno")
            sigma_channels: Number of conductivity channels (6 for symmetric tensor)
            source_channels: Number of source field channels
            coord_channels: Number of coordinate channels (3 for X, Y, Z)
            out_channels: Number of output channels (1 for potential)
            geometry_hidden_dim: Hidden dimension for geometry encoder
            geometry_num_layers: Number of layers in geometry encoder
            unet_config: Configuration dict for UNet backbone
            fno_config: Configuration dict for FNO backbone
            add_analytical_solution: If True, expect analytical solution as extra input
            use_spacing_conditioning: If True, apply spacing-based conditioning
            spacing_transform: Transform for spacing ("none", "log", "normalized")
            reference_spacing: Reference spacing for normalization (default: 2.0mm)
            residual_learning: If True, predict (u - analytical) and add analytical back
        """
        super().__init__()

        self.backbone_type = backbone_type.lower()
        self.add_analytical_solution = add_analytical_solution
        self.use_spacing_conditioning = use_spacing_conditioning
        self.residual_learning = residual_learning

        # Residual learning requires analytical solution
        if residual_learning and not add_analytical_solution:
            raise ValueError("Residual learning requires add_analytical_solution=True")

        # If using analytical solution, increase source channels
        effective_source_channels = source_channels + (1 if add_analytical_solution else 0)

        # Combined encoder with optional spacing conditioning
        encoder_out_channels = geometry_hidden_dim
        self.encoder = CombinedEncoder(
            sigma_channels=sigma_channels,
            source_channels=effective_source_channels,
            coord_channels=coord_channels,
            geometry_hidden_dim=geometry_hidden_dim,
            geometry_num_layers=geometry_num_layers,
            out_channels=encoder_out_channels,
            use_spacing_conditioning=use_spacing_conditioning,
            spacing_transform=spacing_transform,
            reference_spacing=reference_spacing,
        )

        # Initialize backbone based on type
        if self.backbone_type == "unet":
            unet_config = unet_config or {}
            self.backbone = UNetBackbone(
                in_channels=encoder_out_channels,
                out_channels=out_channels,
                base_channels=unet_config.get("base_channels", 32),
                depth=unet_config.get("depth", 4),
                dropout=unet_config.get("dropout", 0.1),
            )
        elif self.backbone_type == "fno":
            fno_config = fno_config or {}
            self.backbone = FNOBackbone(
                in_channels=encoder_out_channels,
                out_channels=out_channels,
                modes1=fno_config.get("modes1", 8),
                modes2=fno_config.get("modes2", 8),
                modes3=fno_config.get("modes3", 8),
                width=fno_config.get("width", 32),
                num_layers=fno_config.get("num_layers", 4),
                fc_dim=fno_config.get("fc_dim", 128),
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}. Must be 'unet' or 'fno'.")

    def forward(
        self,
        sigma: torch.Tensor,
        source: torch.Tensor,
        coords: torch.Tensor,
        spacing: torch.Tensor,
        analytical: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            sigma: Conductivity tensor (B, 6, D, H, W)
            source: Source field (B, 1, D, H, W)
            coords: Normalized coordinates (B, 3, D, H, W)
            spacing: Voxel spacing (B, 3)
            analytical: Optional analytical solution (B, 1, D, H, W)
            **kwargs: Additional arguments (ignored)

        Returns:
            Predicted potential field (B, 1, D, H, W)
        """
        # Concatenate analytical solution with source if provided
        if self.add_analytical_solution and analytical is not None:
            source = torch.cat([source, analytical], dim=1)

        # Encode inputs
        features = self.encoder(sigma, source, coords, spacing)

        # Predict potential (or residual if residual_learning is enabled)
        output = self.backbone(features)

        # If residual learning, add analytical solution back
        if self.residual_learning and analytical is not None:
            output = output + analytical

        return output


def build_model(config: Dict) -> PotentialFieldModel:
    """Build model from configuration dictionary.

    Args:
        config: Full configuration dictionary

    Returns:
        Initialized PotentialFieldModel
    """
    model_config = config.get("model", {})

    backbone_type = model_config.get("backbone", "fno")

    # Geometry encoder config
    geom_config = model_config.get("geometry_encoder", {})
    geometry_hidden_dim = geom_config.get("hidden_dim", 64)
    geometry_num_layers = geom_config.get("num_layers", 2)

    # Backbone configs
    unet_config = model_config.get("unet", {})
    fno_config = model_config.get("fno", {})

    # Analytical solution option
    add_analytical_solution = model_config.get("add_analytical_solution", False)

    # Residual learning option
    residual_learning = model_config.get("residual_learning", False)

    # Spacing conditioning options (defaults to True for backward compatibility)
    spacing_config = config.get("spacing", {})
    use_spacing_conditioning = spacing_config.get("use_spacing_conditioning", True)
    spacing_transform = spacing_config.get("spacing_transform", "none")
    reference_spacing = spacing_config.get("reference_spacing", 2.0)

    model = PotentialFieldModel(
        backbone_type=backbone_type,
        sigma_channels=6,
        source_channels=1,
        coord_channels=3,
        out_channels=1,
        geometry_hidden_dim=geometry_hidden_dim,
        geometry_num_layers=geometry_num_layers,
        unet_config=unet_config,
        fno_config=fno_config,
        add_analytical_solution=add_analytical_solution,
        use_spacing_conditioning=use_spacing_conditioning,
        spacing_transform=spacing_transform,
        reference_spacing=reference_spacing,
        residual_learning=residual_learning,
    )

    return model


def get_device(config: Optional[Dict] = None) -> torch.device:
    """Get the appropriate device based on availability and config.

    Args:
        config: Optional config dict with device preferences

    Returns:
        torch.device for computation
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
