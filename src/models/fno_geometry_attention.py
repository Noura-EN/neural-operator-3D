"""FNO with Geometry Cross-Attention.

Inspired by GINOT (Geometry-Informed Neural Operator Transformer), this module
combines FNO's spectral convolutions with cross-attention to geometry features.

Key idea: After each FNO layer's spectral processing, the features attend to
geometry tokens, allowing the model to dynamically query relevant geometric
information at each spatial location.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class GeometryTokenEncoder(nn.Module):
    """Encodes 3D geometry (conductivity + mask) into a fixed set of tokens.

    Produces Key and Value matrices for cross-attention with FNO features.
    Uses adaptive pooling to create a fixed number of geometry tokens regardless
    of input resolution.
    """

    def __init__(
        self,
        in_channels: int = 7,  # 6 conductivity + 1 mask
        hidden_dim: int = 64,
        token_dim: int = 64,
        num_tokens: Tuple[int, int, int] = (6, 6, 12),  # D, H, W tokens
        num_layers: int = 3,
    ):
        """Initialize geometry token encoder.

        Args:
            in_channels: Input channels (conductivity tensor + optional mask)
            hidden_dim: Hidden dimension for CNN layers
            token_dim: Dimension of output tokens (for K, V)
            num_tokens: Number of tokens in each dimension (D, H, W)
            num_layers: Number of CNN layers before pooling
        """
        super().__init__()

        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.total_tokens = num_tokens[0] * num_tokens[1] * num_tokens[2]

        # CNN to process geometry
        layers = []
        layers.append(nn.Conv3d(in_channels, hidden_dim, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm3d(hidden_dim))
        layers.append(nn.GELU())

        for _ in range(num_layers - 1):
            layers.append(nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm3d(hidden_dim))
            layers.append(nn.GELU())

        self.cnn = nn.Sequential(*layers)

        # Adaptive pooling to fixed token grid
        self.pool = nn.AdaptiveAvgPool3d(num_tokens)

        # Project to K and V
        self.to_k = nn.Linear(hidden_dim, token_dim)
        self.to_v = nn.Linear(hidden_dim, token_dim)

        # Learnable positional encoding for tokens
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.total_tokens, token_dim) * 0.02
        )

    def forward(
        self,
        sigma: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode geometry into K, V tokens.

        Args:
            sigma: Conductivity tensor (B, 6, D, H, W)
            mask: Optional muscle mask (B, 1, D, H, W)

        Returns:
            K: Key matrix (B, num_tokens, token_dim)
            V: Value matrix (B, num_tokens, token_dim)
        """
        # Concatenate inputs
        if mask is not None:
            x = torch.cat([sigma, mask], dim=1)
        else:
            x = sigma

        # CNN processing
        x = self.cnn(x)  # (B, hidden_dim, D, H, W)

        # Pool to fixed token grid
        x = self.pool(x)  # (B, hidden_dim, num_tokens[0], num_tokens[1], num_tokens[2])

        # Reshape to tokens
        B = x.shape[0]
        x = x.permute(0, 2, 3, 4, 1)  # (B, D, H, W, hidden_dim)
        x = x.reshape(B, self.total_tokens, -1)  # (B, num_tokens, hidden_dim)

        # Project to K, V and add positional encoding
        K = self.to_k(x) + self.pos_encoding
        V = self.to_v(x) + self.pos_encoding

        return K, V


class GeometryCrossAttention(nn.Module):
    """Cross-attention between FNO features and geometry tokens.

    FNO features (queries) attend to geometry tokens (keys, values).
    Uses multi-head attention with optional chunking for memory efficiency.
    """

    def __init__(
        self,
        feature_dim: int,
        token_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        chunk_size: int = 4096,  # Process queries in chunks for memory
    ):
        """Initialize cross-attention.

        Args:
            feature_dim: Dimension of FNO features (query dim)
            token_dim: Dimension of geometry tokens (key/value dim)
            num_heads: Number of attention heads
            dropout: Attention dropout
            chunk_size: Number of queries to process at once (for memory)
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.chunk_size = chunk_size
        self.scale = self.head_dim ** -0.5

        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        # Query projection (from FNO features)
        self.q_proj = nn.Linear(feature_dim, feature_dim)

        # Key, Value projections (from geometry tokens, project to feature_dim)
        self.k_proj = nn.Linear(token_dim, feature_dim)
        self.v_proj = nn.Linear(token_dim, feature_dim)

        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        self.dropout = nn.Dropout(dropout)

        # Layer norm for stability
        self.norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        features: torch.Tensor,
        geom_k: torch.Tensor,
        geom_v: torch.Tensor,
    ) -> torch.Tensor:
        """Apply cross-attention.

        Args:
            features: FNO features (B, C, D, H, W)
            geom_k: Geometry keys (B, num_tokens, token_dim)
            geom_v: Geometry values (B, num_tokens, token_dim)

        Returns:
            Attended features (B, C, D, H, W)
        """
        B, C, D, H, W = features.shape
        N = D * H * W  # Number of spatial positions

        # Reshape features to (B, N, C)
        x = features.permute(0, 2, 3, 4, 1).reshape(B, N, C)

        # Project K, V from geometry tokens
        K = self.k_proj(geom_k)  # (B, num_tokens, feature_dim)
        V = self.v_proj(geom_v)  # (B, num_tokens, feature_dim)

        # Reshape K, V for multi-head attention
        num_tokens = K.shape[1]
        K = K.view(B, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        # K, V: (B, num_heads, num_tokens, head_dim)

        # Process queries in chunks for memory efficiency
        if N <= self.chunk_size:
            # Small enough to process all at once
            out = self._attention(x, K, V)
        else:
            # Process in chunks
            out_chunks = []
            for i in range(0, N, self.chunk_size):
                chunk = x[:, i:i+self.chunk_size, :]
                out_chunk = self._attention(chunk, K, V)
                out_chunks.append(out_chunk)
            out = torch.cat(out_chunks, dim=1)

        # Residual connection with pre-norm
        out = x + self.dropout(out)
        out = self.norm(out)

        # Reshape back to (B, C, D, H, W)
        out = out.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

        return out

    def _attention(
        self,
        x: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-head attention for a chunk of queries.

        Args:
            x: Query features (B, chunk_size, feature_dim)
            K: Keys (B, num_heads, num_tokens, head_dim)
            V: Values (B, num_heads, num_tokens, head_dim)

        Returns:
            Attended features (B, chunk_size, feature_dim)
        """
        B, chunk_size, _ = x.shape

        # Project queries
        Q = self.q_proj(x)  # (B, chunk_size, feature_dim)
        Q = Q.view(B, chunk_size, self.num_heads, self.head_dim).transpose(1, 2)
        # Q: (B, num_heads, chunk_size, head_dim)

        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # attn: (B, num_heads, chunk_size, num_tokens)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Attend to values
        out = torch.matmul(attn, V)
        # out: (B, num_heads, chunk_size, head_dim)

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, chunk_size, self.feature_dim)
        out = self.out_proj(out)

        return out


class SpectralConv3d(nn.Module):
    """3D Spectral Convolution layer using real FFT.

    (Same as in fno.py - copied here for self-contained module)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)

        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights3 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights4 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )

    def compl_mul3d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        D, H, W = x.shape[2], x.shape[3], x.shape[4]

        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))

        out_ft = torch.zeros(
            batch_size, self.out_channels,
            D, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2
        )
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4
        )

        x = torch.fft.irfftn(out_ft, s=(D, H, W), dim=(-3, -2, -1))
        return x


class FNOBlockWithGeometryAttention(nn.Module):
    """FNO block with geometry cross-attention.

    Structure:
        x = SpectralConv(x) + Linear(x)    # Standard FNO
        x = CrossAttention(x, geom_K, geom_V)  # Geometry attention
        x = Activation(x)
    """

    def __init__(
        self,
        width: int,
        modes1: int,
        modes2: int,
        modes3: int,
        token_dim: int,
        num_heads: int = 4,
        attention_dropout: float = 0.0,
    ):
        super().__init__()

        self.spectral_conv = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.linear = nn.Conv3d(width, width, kernel_size=1)

        self.cross_attention = GeometryCrossAttention(
            feature_dim=width,
            token_dim=token_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )

        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        geom_k: torch.Tensor,
        geom_v: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with geometry attention.

        Args:
            x: Input features (B, width, D, H, W)
            geom_k: Geometry keys (B, num_tokens, token_dim)
            geom_v: Geometry values (B, num_tokens, token_dim)

        Returns:
            Output features (B, width, D, H, W)
        """
        # Spectral + linear path
        x1 = self.spectral_conv(x)
        x2 = self.linear(x)
        x = x1 + x2

        # Geometry cross-attention
        x = self.cross_attention(x, geom_k, geom_v)

        # Activation
        x = self.activation(x)

        return x


class FNOWithGeometryAttention(nn.Module):
    """FNO with geometry cross-attention after each layer.

    This architecture combines:
    1. FNO's spectral convolutions for learning global patterns in Fourier space
    2. Cross-attention to geometry tokens for local/boundary-aware processing

    Inspired by GINOT (Geometry-Informed Neural Operator Transformer).
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 1,
        modes1: int = 8,
        modes2: int = 8,
        modes3: int = 8,
        width: int = 32,
        num_layers: int = 4,
        fc_dim: int = 128,
        # Geometry attention params
        geometry_in_channels: int = 7,  # 6 conductivity + 1 mask
        geometry_hidden_dim: int = 64,
        geometry_token_dim: int = 64,
        geometry_num_tokens: Tuple[int, int, int] = (6, 6, 12),
        geometry_num_layers: int = 3,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.0,
    ):
        """Initialize FNO with geometry attention.

        Args:
            in_channels: Input channels from encoder
            out_channels: Output channels (1 for potential)
            modes1, modes2, modes3: Fourier modes in each dimension
            width: FNO hidden width
            num_layers: Number of FNO layers
            fc_dim: FC decoder dimension
            geometry_in_channels: Channels for geometry input
            geometry_hidden_dim: Hidden dim for geometry encoder CNN
            geometry_token_dim: Dimension of geometry tokens
            geometry_num_tokens: Number of tokens in (D, H, W)
            geometry_num_layers: CNN layers in geometry encoder
            num_attention_heads: Attention heads for cross-attention
            attention_dropout: Dropout in attention
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.num_layers = num_layers

        # Geometry token encoder
        self.geometry_encoder = GeometryTokenEncoder(
            in_channels=geometry_in_channels,
            hidden_dim=geometry_hidden_dim,
            token_dim=geometry_token_dim,
            num_tokens=geometry_num_tokens,
            num_layers=geometry_num_layers,
        )

        # Lifting layer
        self.lift = nn.Conv3d(in_channels, width, kernel_size=1)

        # FNO layers with geometry attention
        self.fno_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.fno_layers.append(
                FNOBlockWithGeometryAttention(
                    width=width,
                    modes1=modes1,
                    modes2=modes2,
                    modes3=modes3,
                    token_dim=geometry_token_dim,
                    num_heads=num_attention_heads,
                    attention_dropout=attention_dropout,
                )
            )

        # Projection layers (decoder)
        self.proj1 = nn.Conv3d(width, fc_dim, kernel_size=1)
        self.proj2 = nn.Conv3d(fc_dim, out_channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(
        self,
        features: torch.Tensor,
        sigma: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Encoded features from CombinedEncoder (B, in_channels, D, H, W)
            sigma: Conductivity tensor (B, 6, D, H, W) - for geometry encoding
            mask: Optional muscle mask (B, 1, D, H, W) - for geometry encoding

        Returns:
            Predicted potential (B, out_channels, D, H, W)
        """
        # Encode geometry into K, V tokens
        geom_k, geom_v = self.geometry_encoder(sigma, mask)

        # Lifting
        x = self.lift(features)

        # FNO layers with geometry attention
        for layer in self.fno_layers:
            x = layer(x, geom_k, geom_v)

        # Projection (decode)
        x = self.activation(self.proj1(x))
        x = self.proj2(x)

        return x


class FNOGeometryAttentionBackbone(nn.Module):
    """Backbone wrapper for FNO with geometry attention."""

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 1,
        modes1: int = 8,
        modes2: int = 8,
        modes3: int = 8,
        width: int = 32,
        num_layers: int = 4,
        fc_dim: int = 128,
        geometry_config: Optional[dict] = None,
    ):
        """Initialize backbone.

        Args:
            in_channels: Input channels from encoder
            out_channels: Output channels
            modes1, modes2, modes3: Fourier modes
            width: FNO width
            num_layers: Number of FNO layers
            fc_dim: FC dimension
            geometry_config: Dict with geometry attention params
        """
        super().__init__()

        geometry_config = geometry_config or {}

        self.fno = FNOWithGeometryAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            modes1=modes1,
            modes2=modes2,
            modes3=modes3,
            width=width,
            num_layers=num_layers,
            fc_dim=fc_dim,
            geometry_in_channels=geometry_config.get("in_channels", 7),
            geometry_hidden_dim=geometry_config.get("hidden_dim", 64),
            geometry_token_dim=geometry_config.get("token_dim", 64),
            geometry_num_tokens=tuple(geometry_config.get("num_tokens", [6, 6, 12])),
            geometry_num_layers=geometry_config.get("num_layers", 3),
            num_attention_heads=geometry_config.get("num_heads", 4),
            attention_dropout=geometry_config.get("dropout", 0.0),
        )

        # Store for reference
        self.needs_geometry = True

    def forward(
        self,
        features: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Encoded features (B, in_channels, D, H, W)
            sigma: Conductivity tensor (B, 6, D, H, W)
            mask: Optional muscle mask (B, 1, D, H, W)

        Returns:
            Predicted potential (B, 1, D, H, W)
        """
        if sigma is None:
            raise ValueError("FNOGeometryAttentionBackbone requires sigma for geometry encoding")

        return self.fno(features, sigma, mask)
