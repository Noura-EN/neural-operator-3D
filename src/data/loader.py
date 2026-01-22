"""Data loading utilities for 3D potential field prediction."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PotentialFieldDataset(Dataset):
    """Dataset for loading 3D potential field data from .npz files.

    Each sample contains:
        - sigma: Conductivity tensor (6 channels: 3 diagonal + 3 off-diagonal)
        - source: Source field f
        - mask: Binary mask for valid regions
        - u: FEM ground truth potential
        - spacing: Physical voxel spacing
        - source_point: Location of the source
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        sample_indices: Optional[List[int]] = None,
        coord_range: Tuple[float, float] = (-1.0, 1.0),
        device: str = "cpu",
        use_physical_coords: bool = False,
        add_spacing_channels: bool = False,
    ):
        """Initialize dataset.

        Args:
            data_dir: Directory containing .npz sample files
            sample_indices: List of sample indices to use (for train/val/test splits)
            coord_range: Range for normalized coordinates (default: [-1, 1])
            device: Device to load tensors to
            use_physical_coords: If True, scale coordinates by spacing
            add_spacing_channels: If True, add spacing as explicit input channels
        """
        self.data_dir = Path(data_dir)
        self.coord_range = coord_range
        self.device = device
        self.use_physical_coords = use_physical_coords
        self.add_spacing_channels = add_spacing_channels

        # Find all sample files
        all_files = sorted(self.data_dir.glob("sample_*.npz"))

        if sample_indices is not None:
            self.sample_files = [all_files[i] for i in sample_indices if i < len(all_files)]
        else:
            self.sample_files = all_files

        if len(self.sample_files) == 0:
            raise ValueError(f"No sample files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.sample_files)

    def _generate_coords(
        self,
        shape: Tuple[int, int, int],
        spacing: Optional[np.ndarray] = None,
        use_physical_coords: bool = False,
    ) -> torch.Tensor:
        """Generate coordinate grids.

        Args:
            shape: Grid shape (D, H, W)
            spacing: Physical voxel spacing (3,) - only used if use_physical_coords=True
            use_physical_coords: If True, scale coords by spacing to get physical positions

        Returns:
            Coordinate tensor of shape (3, D, H, W) with X, Y, Z meshgrids
        """
        D, H, W = shape
        low, high = self.coord_range

        # Create 1D coordinate arrays (normalized to coord_range)
        z = torch.linspace(low, high, D)
        y = torch.linspace(low, high, H)
        x = torch.linspace(low, high, W)

        # Create meshgrid with 'ij' indexing
        Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')

        # Stack to (3, D, H, W) - order is X, Y, Z
        coords = torch.stack([X, Y, Z], dim=0)

        # If using physical coordinates, scale by spacing
        if use_physical_coords and spacing is not None:
            # spacing is (dx, dy, dz), coords are (X, Y, Z)
            # Scale each coordinate by corresponding spacing
            # This makes coordinates represent physical positions
            spacing_tensor = torch.from_numpy(spacing).float()
            # Reshape for broadcasting: (3,) -> (3, 1, 1, 1)
            spacing_scale = spacing_tensor.view(3, 1, 1, 1)
            coords = coords * spacing_scale

        return coords

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample.

        Returns:
            Dictionary containing:
                - sigma: (6, D, H, W) conductivity tensor
                - source: (1, D, H, W) source field
                - coords: (3, D, H, W) normalized coordinates (X, Y, Z)
                - spacing: (3,) physical voxel spacing
                - mask: (1, D, H, W) binary mask
                - target: (1, D, H, W) FEM ground truth
                - source_point: (3,) source location
        """
        data = np.load(self.sample_files[idx])

        # Load arrays - data is stored as (D, H, W, C) or (D, H, W)
        sigma = data['sigma']  # (D, H, W, 6)
        source = data['source']  # (D, H, W)
        mask = data['mask']  # (D, H, W)
        u = data['u']  # (D, H, W) - FEM ground truth
        spacing = data['spacing']  # (3,)
        source_point = data['source_point']  # (3,)

        # Convert to torch tensors
        # Transpose sigma from (D, H, W, 6) to (6, D, H, W)
        sigma = torch.from_numpy(sigma).permute(3, 0, 1, 2).float()
        source = torch.from_numpy(source).unsqueeze(0).float()  # (1, D, H, W)
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)  # (1, D, H, W)
        target = torch.from_numpy(u).unsqueeze(0).float()  # (1, D, H, W)
        spacing = torch.from_numpy(spacing).float()  # (3,)
        source_point = torch.from_numpy(source_point).float()  # (3,)

        # Generate coordinates (normalized or physical)
        grid_shape = sigma.shape[1:]  # (D, H, W)
        spacing_np = data['spacing']  # Keep numpy version for coord generation
        coords = self._generate_coords(
            grid_shape,
            spacing=spacing_np,
            use_physical_coords=self.use_physical_coords,
        )

        result = {
            'sigma': sigma,
            'source': source,
            'coords': coords,
            'spacing': spacing,
            'mask': mask,
            'target': target,
            'source_point': source_point,
        }

        # Optionally add spacing as explicit channels
        if self.add_spacing_channels:
            # Broadcast spacing to (3, D, H, W)
            D, H, W = grid_shape
            spacing_channels = spacing.view(3, 1, 1, 1).expand(3, D, H, W)
            result['spacing_channels'] = spacing_channels

        return result


def create_data_splits(
    data_dir: Union[str, Path],
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Create train/val/test splits from data directory.

    Args:
        data_dir: Directory containing sample files
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    data_dir = Path(data_dir)
    all_files = sorted(data_dir.glob("sample_*.npz"))
    n_samples = len(all_files)

    # Set seed for reproducibility
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    # Calculate split sizes
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * val_split)

    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train:n_train + n_val].tolist()
    test_indices = indices[n_train + n_val:].tolist()

    return train_indices, val_indices, test_indices


def get_dataloaders(
    config: dict,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders from config.

    Args:
        config: Configuration dictionary
        seed: Random seed

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_config = config['data']

    # Create splits
    train_indices, val_indices, test_indices = create_data_splits(
        data_dir=data_config['data_dir'],
        train_split=data_config['train_split'],
        val_split=data_config['val_split'],
        test_split=data_config['test_split'],
        seed=seed,
    )

    coord_range = tuple(config['grid']['coord_range'])

    # Spacing encoding options
    spacing_config = config.get('spacing', {})
    use_physical_coords = spacing_config.get('use_physical_coords', False)
    add_spacing_channels = spacing_config.get('add_spacing_channels', False)

    # Create datasets
    train_dataset = PotentialFieldDataset(
        data_dir=data_config['data_dir'],
        sample_indices=train_indices,
        coord_range=coord_range,
        use_physical_coords=use_physical_coords,
        add_spacing_channels=add_spacing_channels,
    )

    val_dataset = PotentialFieldDataset(
        data_dir=data_config['data_dir'],
        sample_indices=val_indices,
        coord_range=coord_range,
        use_physical_coords=use_physical_coords,
        add_spacing_channels=add_spacing_channels,
    )

    test_dataset = PotentialFieldDataset(
        data_dir=data_config['data_dir'],
        sample_indices=test_indices,
        coord_range=coord_range,
        use_physical_coords=use_physical_coords,
        add_spacing_channels=add_spacing_channels,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        drop_last=True if len(train_dataset) > data_config['batch_size'] else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
    )

    return train_loader, val_loader, test_loader
