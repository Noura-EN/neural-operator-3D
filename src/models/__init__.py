"""Models module for 3D potential field prediction."""

from .geometry import GeometryEncoder, SpacingConditioner, CombinedEncoder
from .unet import UNet3D, UNetBackbone
from .fno import SpectralConv3d, FNOBlock, FNO3D, FNOBackbone
from .tfno import FactorizedSpectralConv3d, TFNOBlock, TFNO3D, TFNOBackbone
from .uno import UNOEncoderBlock, UNODecoderBlock, UNO3D, UNOBackbone
from .deeponet import BranchNet3D, TrunkNet3D, DeepONet3D, DeepONetBackbone
from .lsm import LatentSpectralConv3d, LSM3D, LSMBackbone
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
    # U-NO (U-shaped Neural Operator)
    "UNOEncoderBlock",
    "UNODecoderBlock",
    "UNO3D",
    "UNOBackbone",
    # DeepONet
    "BranchNet3D",
    "TrunkNet3D",
    "DeepONet3D",
    "DeepONetBackbone",
    # LSM (Latent Spectral Model)
    "LatentSpectralConv3d",
    "LSM3D",
    "LSMBackbone",
    # Wrapper
    "PotentialFieldModel",
    "build_model",
    "get_device",
    "count_parameters",
]
