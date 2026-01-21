"""Models module for 3D potential field prediction."""

from .geometry import GeometryEncoder, SpacingConditioner, CombinedEncoder
from .unet import UNet3D, UNetBackbone
from .fno import SpectralConv3d, FNOBlock, FNO3D, FNOBackbone
from .wrapper import PotentialFieldModel, build_model, get_device, count_parameters

__all__ = [
    "GeometryEncoder",
    "SpacingConditioner",
    "CombinedEncoder",
    "UNet3D",
    "UNetBackbone",
    "SpectralConv3d",
    "FNOBlock",
    "FNO3D",
    "FNOBackbone",
    "PotentialFieldModel",
    "build_model",
    "get_device",
    "count_parameters",
]
