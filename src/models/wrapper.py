"""Model wrapper that combines encoder and backbone for potential field prediction."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .geometry import CombinedEncoder
from .unet import UNetBackbone
from .fno import FNOBackbone
from .tfno import TFNOBackbone
from .uno import UNOBackbone
from .deeponet import DeepONetBackbone
from .lsm import LSMBackbone
from .fno_geometry_attention import FNOGeometryAttentionBackbone
from .fno_geometry_attention_lite import FNOGeometryAttentionLiteBackbone


class PotentialFieldModel(nn.Module):
    """Unified model for potential field prediction.

    Combines:
    1. CombinedEncoder: Processes conductivity, source, and coordinates
       with additive spacing conditioning
    2. Backbone: FNO, TFNO, U-NO, DeepONet, LSM, or UNet for predicting the potential field
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
        geometry_use_residual: bool = False,
        unet_config: Optional[Dict] = None,
        fno_config: Optional[Dict] = None,
        uno_config: Optional[Dict] = None,
        deeponet_config: Optional[Dict] = None,
        lsm_config: Optional[Dict] = None,
        geometry_attention_config: Optional[Dict] = None,
        add_analytical_solution: bool = False,
        add_distance_field: bool = False,
        use_spacing_conditioning: bool = True,
    ):
        """Initialize the potential field model.

        Args:
            backbone_type: Type of backbone ("unet", "fno", "tfno", "uno", "deeponet", "lsm", "fno_geom_attn")
            sigma_channels: Number of conductivity channels (6 for symmetric tensor)
            source_channels: Number of source field channels
            coord_channels: Number of coordinate channels (3 for X, Y, Z)
            out_channels: Number of output channels (1 for potential)
            geometry_hidden_dim: Hidden dimension for geometry encoder
            geometry_num_layers: Number of layers in geometry encoder
            geometry_use_residual: If True, use residual blocks in geometry encoder
            unet_config: Configuration dict for UNet backbone
            fno_config: Configuration dict for FNO/TFNO backbone
            uno_config: Configuration dict for U-NO backbone
            deeponet_config: Configuration dict for DeepONet backbone
            lsm_config: Configuration dict for LSM backbone
            geometry_attention_config: Configuration dict for FNO with geometry attention
            add_analytical_solution: If True, expect analytical solution as extra input
            add_distance_field: If True, expect distance-to-boundary field as extra input
            use_spacing_conditioning: If True, apply spacing-based conditioning
        """
        super().__init__()

        self.backbone_type = backbone_type.lower()
        self.add_analytical_solution = add_analytical_solution
        self.add_distance_field = add_distance_field
        self.use_spacing_conditioning = use_spacing_conditioning

        # If using analytical solution, increase source channels
        effective_source_channels = source_channels + (1 if add_analytical_solution else 0)

        # Combined encoder with optional spacing conditioning
        encoder_out_channels = geometry_hidden_dim
        self.encoder = CombinedEncoder(
            sigma_channels=sigma_channels,
            source_channels=effective_source_channels,
            coord_channels=coord_channels,
            distance_channels=1 if add_distance_field else 0,
            geometry_hidden_dim=geometry_hidden_dim,
            geometry_num_layers=geometry_num_layers,
            geometry_use_residual=geometry_use_residual,
            out_channels=encoder_out_channels,
            use_spacing_conditioning=use_spacing_conditioning,
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
                num_layers=fno_config.get("num_layers", 6),
                fc_dim=fno_config.get("fc_dim", 128),
            )
        elif self.backbone_type == "tfno":
            fno_config = fno_config or {}
            self.backbone = TFNOBackbone(
                in_channels=encoder_out_channels,
                out_channels=out_channels,
                modes1=fno_config.get("modes1", 8),
                modes2=fno_config.get("modes2", 8),
                modes3=fno_config.get("modes3", 8),
                width=fno_config.get("width", 32),
                num_layers=fno_config.get("num_layers", 6),
                fc_dim=fno_config.get("fc_dim", 128),
            )
        elif self.backbone_type == "uno":
            uno_config = uno_config or {}
            self.backbone = UNOBackbone(
                in_channels=encoder_out_channels,
                out_channels=out_channels,
                base_width=uno_config.get("base_width", 32),
                depth=uno_config.get("depth", 3),
                base_modes=uno_config.get("base_modes", 8),
                fc_dim=uno_config.get("fc_dim", 128),
            )
        elif self.backbone_type == "deeponet":
            deeponet_config = deeponet_config or {}
            self.backbone = DeepONetBackbone(
                in_channels=encoder_out_channels,
                out_channels=out_channels,
                hidden_dim=deeponet_config.get("hidden_dim", 128),
                num_basis=deeponet_config.get("num_basis", 64),
                branch_layers=deeponet_config.get("branch_layers", 4),
                trunk_layers=deeponet_config.get("trunk_layers", 4),
            )
        elif self.backbone_type == "lsm":
            lsm_config = lsm_config or {}
            self.backbone = LSMBackbone(
                in_channels=encoder_out_channels,
                out_channels=out_channels,
                latent_dim=lsm_config.get("latent_dim", 32),
                latent_resolution=tuple(lsm_config.get("latent_resolution", [12, 6, 6])),
                num_layers=lsm_config.get("num_layers", 4),
                hidden_dim=lsm_config.get("hidden_dim", 64),
                latent_modes=tuple(lsm_config.get("latent_modes", [6, 3, 3])),
            )
        elif self.backbone_type == "fno_geom_attn":
            # FNO with geometry cross-attention (inspired by GINOT)
            fno_config = fno_config or {}
            geometry_attention_config = geometry_attention_config or {}
            self.backbone = FNOGeometryAttentionBackbone(
                in_channels=encoder_out_channels,
                out_channels=out_channels,
                modes1=fno_config.get("modes1", 8),
                modes2=fno_config.get("modes2", 8),
                modes3=fno_config.get("modes3", 8),
                width=fno_config.get("width", 32),
                num_layers=fno_config.get("num_layers", 6),
                fc_dim=fno_config.get("fc_dim", 128),
                geometry_config=geometry_attention_config,
            )
        elif self.backbone_type == "fno_geom_attn_lite":
            # Lightweight FNO with geometry attention (~2-3GB vs 12GB)
            fno_config = fno_config or {}
            geometry_attention_config = geometry_attention_config or {}
            self.backbone = FNOGeometryAttentionLiteBackbone(
                in_channels=encoder_out_channels,
                out_channels=out_channels,
                modes1=fno_config.get("modes1", 8),
                modes2=fno_config.get("modes2", 8),
                modes3=fno_config.get("modes3", 8),
                width=fno_config.get("width", 32),
                num_layers=fno_config.get("num_layers", 6),
                fc_dim=fno_config.get("fc_dim", 128),
                geometry_config=geometry_attention_config,
            )
        else:
            raise ValueError(
                f"Unknown backbone type: {backbone_type}. "
                "Must be 'unet', 'fno', 'tfno', 'uno', 'deeponet', 'lsm', 'fno_geom_attn', or 'fno_geom_attn_lite'."
            )

    def forward(
        self,
        sigma: torch.Tensor,
        source: torch.Tensor,
        coords: torch.Tensor,
        spacing: torch.Tensor,
        analytical: Optional[torch.Tensor] = None,
        distance_field: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            sigma: Conductivity tensor (B, 6, D, H, W)
            source: Source field (B, 1, D, H, W)
            coords: Normalized coordinates (B, 3, D, H, W)
            spacing: Voxel spacing (B, 3)
            analytical: Optional analytical solution (B, 1, D, H, W)
            distance_field: Optional signed distance to boundary (B, 1, D, H, W)
            mask: Optional muscle mask (B, 1, D, H, W) for geometry attention
            **kwargs: Additional arguments (ignored)

        Returns:
            Predicted potential field (B, 1, D, H, W)
        """
        # Concatenate analytical solution with source if provided
        if self.add_analytical_solution and analytical is not None:
            source = torch.cat([source, analytical], dim=1)

        # Encode inputs (pass distance field if using it)
        dist = distance_field if self.add_distance_field else None
        features = self.encoder(sigma, source, coords, spacing, distance_field=dist)

        # Predict potential
        # For geometry attention backbone, pass sigma and mask
        if hasattr(self.backbone, 'needs_geometry') and self.backbone.needs_geometry:
            output = self.backbone(features, sigma=sigma, mask=mask)
        else:
            output = self.backbone(features)

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
    geometry_use_residual = geom_config.get("use_residual", False)

    # Backbone configs
    unet_config = model_config.get("unet", {})
    fno_config = model_config.get("fno", {})
    uno_config = model_config.get("uno", {})
    deeponet_config = model_config.get("deeponet", {})
    lsm_config = model_config.get("lsm", {})
    geometry_attention_config = model_config.get("geometry_attention", {})

    # Analytical solution option
    add_analytical_solution = model_config.get("add_analytical_solution", False)

    # Distance field option
    add_distance_field = model_config.get("add_distance_field", False)

    # Spacing conditioning (defaults to True - the best approach)
    use_spacing_conditioning = config.get("spacing", {}).get("use_spacing_conditioning", True)

    model = PotentialFieldModel(
        backbone_type=backbone_type,
        sigma_channels=6,
        source_channels=1,
        coord_channels=3,
        out_channels=1,
        geometry_hidden_dim=geometry_hidden_dim,
        geometry_num_layers=geometry_num_layers,
        geometry_use_residual=geometry_use_residual,
        unet_config=unet_config,
        fno_config=fno_config,
        uno_config=uno_config,
        deeponet_config=deeponet_config,
        lsm_config=lsm_config,
        geometry_attention_config=geometry_attention_config,
        add_analytical_solution=add_analytical_solution,
        add_distance_field=add_distance_field,
        use_spacing_conditioning=use_spacing_conditioning,
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
