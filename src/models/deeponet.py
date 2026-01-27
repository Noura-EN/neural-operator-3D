"""DeepONet (Deep Operator Network) for 3D potential field prediction.

DeepONet uses a branch-trunk architecture where:
- Branch network: Encodes the input function (conductivity, source)
- Trunk network: Encodes the query coordinates
- Output: Dot product of branch and trunk outputs

Reference: Lu et al., "Learning nonlinear operators via DeepONet" (Nature MI 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BranchNet3D(nn.Module):
    """Branch network that encodes the input functions.

    Processes conductivity and source fields using 3D convolutions
    followed by global pooling to produce basis coefficients.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        num_basis: int = 64,
        num_layers: int = 4,
    ):
        """Initialize branch network.

        Args:
            in_channels: Input channels (from encoder)
            hidden_dim: Hidden layer dimension
            num_basis: Number of basis functions (output dimension)
            num_layers: Number of convolutional layers
        """
        super().__init__()

        self.num_basis = num_basis

        # Convolutional encoder
        layers = []
        ch = in_channels
        for i in range(num_layers):
            out_ch = hidden_dim * (2 ** min(i, 2))  # Cap channel growth
            layers.extend([
                nn.Conv3d(ch, out_ch, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.GELU(),
            ])
            if i < num_layers - 1:
                layers.append(nn.AvgPool3d(2))  # Downsample
            ch = out_ch

        self.conv_encoder = nn.Sequential(*layers)

        # Global pooling and projection to basis coefficients
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_basis),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input functions to basis coefficients.

        Args:
            x: Input features (B, C, D, H, W)

        Returns:
            Basis coefficients (B, num_basis)
        """
        x = self.conv_encoder(x)
        x = self.global_pool(x).squeeze(-1).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x


class TrunkNet3D(nn.Module):
    """Trunk network that encodes query coordinates.

    Processes normalized coordinates through an MLP to produce
    basis function evaluations at each spatial location.
    """

    def __init__(
        self,
        coord_dim: int = 3,
        hidden_dim: int = 128,
        num_basis: int = 64,
        num_layers: int = 4,
    ):
        """Initialize trunk network.

        Args:
            coord_dim: Coordinate dimensions (3 for 3D)
            hidden_dim: Hidden layer dimension
            num_basis: Number of basis functions
            num_layers: Number of MLP layers
        """
        super().__init__()

        self.num_basis = num_basis

        # MLP for coordinate encoding
        layers = []
        ch = coord_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(ch, hidden_dim),
                nn.GELU(),
            ])
            ch = hidden_dim

        layers.append(nn.Linear(ch, num_basis))
        self.mlp = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Evaluate basis functions at coordinates.

        Args:
            coords: Coordinates (B, 3, D, H, W)

        Returns:
            Basis evaluations (B, num_basis, D, H, W)
        """
        B, C, D, H, W = coords.shape

        # Reshape for MLP: (B, D, H, W, 3)
        coords_flat = coords.permute(0, 2, 3, 4, 1)

        # Apply MLP at each point
        basis = self.mlp(coords_flat)  # (B, D, H, W, num_basis)

        # Reshape back: (B, num_basis, D, H, W)
        basis = basis.permute(0, 4, 1, 2, 3)

        return basis


class DeepONet3D(nn.Module):
    """3D DeepONet for potential field prediction.

    The output at each point is the dot product of branch outputs
    (basis coefficients) and trunk outputs (basis evaluations).
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 1,
        hidden_dim: int = 128,
        num_basis: int = 64,
        branch_layers: int = 4,
        trunk_layers: int = 4,
    ):
        """Initialize DeepONet.

        Args:
            in_channels: Input feature channels
            out_channels: Output channels
            hidden_dim: Hidden dimension for both networks
            num_basis: Number of basis functions
            branch_layers: Number of layers in branch network
            trunk_layers: Number of layers in trunk network
        """
        super().__init__()

        self.num_basis = num_basis
        self.out_channels = out_channels

        # Branch network (processes input functions)
        self.branch = BranchNet3D(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_basis=num_basis * out_channels,
            num_layers=branch_layers,
        )

        # Trunk network (processes coordinates)
        self.trunk = TrunkNet3D(
            coord_dim=3,
            hidden_dim=hidden_dim,
            num_basis=num_basis,
            num_layers=trunk_layers,
        )

        # Bias term
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through DeepONet.

        Args:
            features: Encoded input features (B, C, D, H, W)
            coords: Coordinates (B, 3, D, H, W)

        Returns:
            Predicted field (B, out_channels, D, H, W)
        """
        B, _, D, H, W = features.shape

        # Branch: input functions -> basis coefficients
        branch_out = self.branch(features)  # (B, num_basis * out_channels)
        branch_out = branch_out.view(B, self.out_channels, self.num_basis)

        # Trunk: coordinates -> basis evaluations
        trunk_out = self.trunk(coords)  # (B, num_basis, D, H, W)

        # Dot product: sum over basis dimension
        # branch_out: (B, out_channels, num_basis)
        # trunk_out: (B, num_basis, D, H, W)
        output = torch.einsum('bcp,bpdhw->bcdhw', branch_out, trunk_out)

        # Add bias
        output = output + self.bias.view(1, -1, 1, 1, 1)

        return output


class DeepONetBackbone(nn.Module):
    """DeepONet backbone wrapper.

    Note: DeepONet requires coordinates as additional input, which
    differs from FNO. The wrapper handles this interface.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 1,
        hidden_dim: int = 128,
        num_basis: int = 64,
        branch_layers: int = 4,
        trunk_layers: int = 4,
    ):
        super().__init__()

        # Store coordinate channels separately
        self.coord_channels = 3
        self.feature_channels = in_channels - self.coord_channels

        self.deeponet = DeepONet3D(
            in_channels=self.feature_channels,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            num_basis=num_basis,
            branch_layers=branch_layers,
            trunk_layers=trunk_layers,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Expects features to have coordinates concatenated as last 3 channels.

        Args:
            features: (B, in_channels, D, H, W) with last 3 channels being coords

        Returns:
            Predicted potential (B, 1, D, H, W)
        """
        # Split features and coordinates
        # Coordinates are the last 3 channels (from CombinedEncoder)
        input_features = features[:, :-self.coord_channels]
        coords = features[:, -self.coord_channels:]

        return self.deeponet(input_features, coords)
