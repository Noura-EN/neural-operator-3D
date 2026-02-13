"""Models module for 3D potential field prediction."""

from .geometry import GeometryEncoder, SpacingConditioner, CombinedEncoder
from .unet import UNet3D, UNetBackbone
from .fno import SpectralConv3d, FNOBlock, FNO3D, FNOBackbone
from .tfno import FactorizedSpectralConv3d, TFNOBlock, TFNO3D, TFNOBackbone
from .fno_geometry_attention_lite import FNOGeometryAttentionLiteBackbone
from .wrapper import PotentialFieldModel, build_model, get_device, count_parameters

__all__ = [
    # Geometry encoding
    "GeometryEncoder",
    "SpacingConditioner",
    "CombinedEncoder",
    # UNet
    "UNet3D",
    "UNetBackbone",
    # FNO
    "SpectralConv3d",
    "FNOBlock",
    "FNO3D",
    "FNOBackbone",
    # TFNO (Factorized FNO)
    "FactorizedSpectralConv3d",
    "TFNOBlock",
    "TFNO3D",
    "TFNOBackbone",
    # FNO with Geometry Attention
    "FNOGeometryAttentionLiteBackbone",
    # Wrapper
    "PotentialFieldModel",
    "build_model",
    "get_device",
    "count_parameters",
]
