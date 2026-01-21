"""Unit tests for FNO spectral convolutions."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.fno import SpectralConv3d, FNOBlock, FNO3D, FNOBackbone


class TestSpectralConv3d:
    """Tests for SpectralConv3d layer."""

    def test_output_shape(self):
        """Test that output shape matches input spatial dimensions."""
        batch_size = 2
        in_channels = 8
        out_channels = 16
        D, H, W = 16, 16, 16
        modes = 4

        layer = SpectralConv3d(in_channels, out_channels, modes, modes, modes)
        x = torch.randn(batch_size, in_channels, D, H, W)

        y = layer(x)

        assert y.shape == (batch_size, out_channels, D, H, W)

    def test_different_spatial_dims(self):
        """Test with non-cubic input dimensions."""
        batch_size = 2
        in_channels = 8
        out_channels = 8
        D, H, W = 32, 16, 8
        modes1, modes2, modes3 = 8, 4, 2

        layer = SpectralConv3d(in_channels, out_channels, modes1, modes2, modes3)
        x = torch.randn(batch_size, in_channels, D, H, W)

        y = layer(x)

        assert y.shape == (batch_size, out_channels, D, H, W)

    def test_gradient_flow(self):
        """Test that gradients flow through the layer."""
        layer = SpectralConv3d(4, 4, 2, 2, 2)
        x = torch.randn(1, 4, 8, 8, 8, requires_grad=True)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_resolution_invariance(self):
        """Test that layer can handle different resolutions."""
        layer = SpectralConv3d(4, 4, 4, 4, 4)

        # Low resolution
        x_low = torch.randn(1, 4, 16, 16, 16)
        y_low = layer(x_low)
        assert y_low.shape == (1, 4, 16, 16, 16)

        # High resolution
        x_high = torch.randn(1, 4, 32, 32, 32)
        y_high = layer(x_high)
        assert y_high.shape == (1, 4, 32, 32, 32)


class TestFNOBlock:
    """Tests for FNO block."""

    def test_output_shape(self):
        """Test FNO block output shape."""
        batch_size = 2
        width = 16
        D, H, W = 16, 16, 16

        block = FNOBlock(width, modes1=4, modes2=4, modes3=4)
        x = torch.randn(batch_size, width, D, H, W)

        y = block(x)

        assert y.shape == x.shape

    def test_nonlinearity(self):
        """Test that FNO block applies nonlinearity."""
        block = FNOBlock(8, modes1=2, modes2=2, modes3=2)
        x = torch.randn(1, 8, 8, 8, 8)

        y = block(x)

        # Output should not be identical to input
        assert not torch.allclose(x, y)


class TestFNO3D:
    """Tests for full FNO3D model."""

    def test_output_shape(self):
        """Test FNO3D output shape."""
        model = FNO3D(
            in_channels=10,
            out_channels=1,
            modes1=4, modes2=4, modes3=4,
            width=16,
            num_layers=2,
            fc_dim=32,
        )

        x = torch.randn(2, 10, 16, 16, 16)
        y = model(x)

        assert y.shape == (2, 1, 16, 16, 16)

    def test_resolution_generalization(self):
        """Test that FNO can handle different resolutions without retraining."""
        model = FNO3D(
            in_channels=10,
            out_channels=1,
            modes1=4, modes2=4, modes3=4,
            width=16,
            num_layers=2,
        )
        model.eval()

        # Train resolution
        x_train = torch.randn(1, 10, 16, 16, 16)
        y_train = model(x_train)
        assert y_train.shape == (1, 1, 16, 16, 16)

        # Higher resolution (zero-shot super-resolution)
        x_high = torch.randn(1, 10, 32, 32, 32)
        y_high = model(x_high)
        assert y_high.shape == (1, 1, 32, 32, 32)

        # Lower resolution
        x_low = torch.randn(1, 10, 8, 8, 8)
        y_low = model(x_low)
        assert y_low.shape == (1, 1, 8, 8, 8)

    def test_gradient_flow(self):
        """Test end-to-end gradient flow."""
        model = FNO3D(
            in_channels=4,
            out_channels=1,
            modes1=2, modes2=2, modes3=2,
            width=8,
            num_layers=1,
        )

        x = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None

        # Check gradients of model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestFNOBackbone:
    """Tests for FNO backbone wrapper."""

    def test_forward(self):
        """Test forward pass."""
        backbone = FNOBackbone(
            in_channels=64,
            out_channels=1,
            modes1=4, modes2=4, modes3=4,
            width=16,
        )

        x = torch.randn(2, 64, 16, 16, 16)
        y = backbone(x)

        assert y.shape == (2, 1, 16, 16, 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
