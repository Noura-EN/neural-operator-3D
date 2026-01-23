"""Data loading utilities for 3D potential field prediction."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


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
        add_analytical_solution: bool = False,
    ):
        """Initialize dataset.

        Args:
            data_dir: Directory containing .npz sample files
            sample_indices: List of sample indices to use (for train/val/test splits)
            coord_range: Range for normalized coordinates (default: [-1, 1])
            device: Device to load tensors to
            use_physical_coords: If True, scale coordinates by spacing
            add_spacing_channels: If True, add spacing as explicit input channels
            add_analytical_solution: If True, compute and add monopole analytical solution
        """
        self.data_dir = Path(data_dir)
        self.coord_range = coord_range
        self.device = device
        self.use_physical_coords = use_physical_coords
        self.add_spacing_channels = add_spacing_channels
        self.add_analytical_solution = add_analytical_solution

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

    def _compute_analytical_solution(
        self,
        sigma: np.ndarray,
        mask: np.ndarray,
        source_point: np.ndarray,
        spacing: np.ndarray,
        grid_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Compute analytical monopole solution Φ = I/(4πσ_avg·r).

        Args:
            sigma: Conductivity tensor (D, H, W, 6) - diagonal components are [0,1,2]
            mask: Binary mask for muscle region (D, H, W)
            source_point: Source location in voxel coordinates (3,)
            spacing: Physical voxel spacing (3,)
            grid_shape: Grid dimensions (D, H, W)

        Returns:
            Analytical solution tensor (1, D, H, W)
        """
        D, H, W = grid_shape

        # Compute σ_avg from muscle region diagonal conductivity
        # sigma has shape (D, H, W, 6) with diagonal at indices 0, 1, 2
        sigma_diag = sigma[..., :3]  # (D, H, W, 3)
        muscle_mask = mask > 0.5

        if muscle_mask.sum() > 0:
            # Average of diagonal components in muscle region
            sigma_avg = sigma_diag[muscle_mask].mean()
        else:
            # Fallback to overall mean
            sigma_avg = sigma_diag.mean()

        # Ensure sigma_avg is positive and reasonable
        sigma_avg = max(sigma_avg, 1e-6)

        # Create coordinate grids in physical space
        z_coords = np.arange(D) * spacing[0]
        y_coords = np.arange(H) * spacing[1]
        x_coords = np.arange(W) * spacing[2]

        Z, Y, X = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')

        # Source position in physical coordinates
        src_z = source_point[0] * spacing[0]
        src_y = source_point[1] * spacing[1]
        src_x = source_point[2] * spacing[2]

        # Compute distance from source
        r = np.sqrt((Z - src_z)**2 + (Y - src_y)**2 + (X - src_x)**2)

        # Avoid division by zero at source point
        r = np.maximum(r, spacing.min())

        # Monopole solution: Φ = I / (4πσr)
        # Assuming unit current I = 1
        analytical = 1.0 / (4.0 * np.pi * sigma_avg * r)

        # Convert to tensor (1, D, H, W)
        return torch.from_numpy(analytical).unsqueeze(0).float()

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

        # Optionally add analytical solution
        if self.add_analytical_solution:
            analytical = self._compute_analytical_solution(
                sigma=data['sigma'],  # Use original numpy array
                mask=data['mask'],
                source_point=data['source_point'],
                spacing=data['spacing'],
                grid_shape=grid_shape,
            )
            result['analytical'] = analytical

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


def create_combined_data_splits(
    data_dirs: List[Union[str, Path]],
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Tuple[Path, int]]]:
    """Create train/val/test splits from multiple data directories.

    This function creates splits where each split contains samples from all directories,
    maintaining the same proportions across directories.

    Args:
        data_dirs: List of directories containing sample files
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        seed: Random seed for reproducibility

    Returns:
        Dict with keys 'train', 'val', 'test', each containing list of (dir, index) tuples
    """
    rng = np.random.default_rng(seed)

    splits = {'train': [], 'val': [], 'test': []}

    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        all_files = sorted(data_dir.glob("sample_*.npz"))
        n_samples = len(all_files)

        if n_samples == 0:
            continue

        # Shuffle indices for this directory
        indices = rng.permutation(n_samples)

        # Calculate split sizes
        n_train = int(n_samples * train_split)
        n_val = int(n_samples * val_split)

        train_indices = indices[:n_train].tolist()
        val_indices = indices[n_train:n_train + n_val].tolist()
        test_indices = indices[n_train + n_val:].tolist()

        # Add (directory, index) tuples to splits
        for idx in train_indices:
            splits['train'].append((data_dir, idx))
        for idx in val_indices:
            splits['val'].append((data_dir, idx))
        for idx in test_indices:
            splits['test'].append((data_dir, idx))

    # Shuffle each split to mix samples from different directories
    for split_name in splits:
        rng.shuffle(splits[split_name])

    return splits


class CombinedPotentialFieldDataset(Dataset):
    """Dataset that combines samples from multiple directories."""

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        coord_range: Tuple[float, float] = (-1.0, 1.0),
        use_physical_coords: bool = False,
        add_spacing_channels: bool = False,
        add_analytical_solution: bool = False,
    ):
        """Initialize combined dataset.

        Args:
            samples: List of (directory, sample_index) tuples
            coord_range: Range for normalized coordinates
            use_physical_coords: If True, scale coordinates by spacing
            add_spacing_channels: If True, add spacing as explicit input channels
            add_analytical_solution: If True, add monopole analytical solution
        """
        self.samples = samples
        self.coord_range = coord_range
        self.use_physical_coords = use_physical_coords
        self.add_spacing_channels = add_spacing_channels
        self.add_analytical_solution = add_analytical_solution

        # Build file list
        self.sample_files = []
        for data_dir, idx in samples:
            all_files = sorted(Path(data_dir).glob("sample_*.npz"))
            if idx < len(all_files):
                self.sample_files.append(all_files[idx])

    def __len__(self) -> int:
        return len(self.sample_files)

    def _generate_coords(
        self,
        shape: Tuple[int, int, int],
        spacing: Optional[np.ndarray] = None,
        use_physical_coords: bool = False,
    ) -> torch.Tensor:
        """Generate coordinate grids."""
        D, H, W = shape
        low, high = self.coord_range

        z = torch.linspace(low, high, D)
        y = torch.linspace(low, high, H)
        x = torch.linspace(low, high, W)

        Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')
        coords = torch.stack([X, Y, Z], dim=0)

        if use_physical_coords and spacing is not None:
            spacing_tensor = torch.from_numpy(spacing).float()
            spacing_scale = spacing_tensor.view(3, 1, 1, 1)
            coords = coords * spacing_scale

        return coords

    def _compute_analytical_solution(
        self,
        sigma: np.ndarray,
        mask: np.ndarray,
        source_point: np.ndarray,
        spacing: np.ndarray,
        grid_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Compute analytical monopole solution."""
        D, H, W = grid_shape
        sigma_diag = sigma[..., :3]
        muscle_mask = mask > 0.5

        if muscle_mask.sum() > 0:
            sigma_avg = sigma_diag[muscle_mask].mean()
        else:
            sigma_avg = sigma_diag.mean()

        sigma_avg = max(sigma_avg, 1e-6)

        z_coords = np.arange(D) * spacing[0]
        y_coords = np.arange(H) * spacing[1]
        x_coords = np.arange(W) * spacing[2]

        Z, Y, X = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')

        src_z = source_point[0] * spacing[0]
        src_y = source_point[1] * spacing[1]
        src_x = source_point[2] * spacing[2]

        r = np.sqrt((Z - src_z)**2 + (Y - src_y)**2 + (X - src_x)**2)
        r = np.maximum(r, spacing.min())

        analytical = 1.0 / (4.0 * np.pi * sigma_avg * r)
        return torch.from_numpy(analytical).unsqueeze(0).float()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample."""
        data = np.load(self.sample_files[idx])

        sigma = data['sigma']
        source = data['source']
        mask = data['mask']
        u = data['u']
        spacing = data['spacing']
        source_point = data['source_point']

        sigma = torch.from_numpy(sigma).permute(3, 0, 1, 2).float()
        source = torch.from_numpy(source).unsqueeze(0).float()
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        target = torch.from_numpy(u).unsqueeze(0).float()
        spacing_tensor = torch.from_numpy(spacing).float()
        source_point_tensor = torch.from_numpy(source_point).float()

        grid_shape = sigma.shape[1:]
        coords = self._generate_coords(
            grid_shape,
            spacing=spacing,
            use_physical_coords=self.use_physical_coords,
        )

        result = {
            'sigma': sigma,
            'source': source,
            'coords': coords,
            'spacing': spacing_tensor,
            'mask': mask,
            'target': target,
            'source_point': source_point_tensor,
        }

        if self.add_spacing_channels:
            D, H, W = grid_shape
            spacing_channels = spacing_tensor.view(3, 1, 1, 1).expand(3, D, H, W)
            result['spacing_channels'] = spacing_channels

        if self.add_analytical_solution:
            analytical = self._compute_analytical_solution(
                sigma=data['sigma'],
                mask=data['mask'],
                source_point=data['source_point'],
                spacing=data['spacing'],
                grid_shape=grid_shape,
            )
            result['analytical'] = analytical

        return result


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

    coord_range = tuple(config['grid']['coord_range'])

    # Spacing encoding options
    spacing_config = config.get('spacing', {})
    use_physical_coords = spacing_config.get('use_physical_coords', False)
    add_spacing_channels = spacing_config.get('add_spacing_channels', False)

    # Analytical solution option
    add_analytical_solution = config.get('model', {}).get('add_analytical_solution', False)

    # Check if using combined directories
    data_dirs = data_config.get('data_dirs', None)

    if data_dirs is not None and len(data_dirs) > 0:
        # Use combined dataset from multiple directories
        splits = create_combined_data_splits(
            data_dirs=data_dirs,
            train_split=data_config['train_split'],
            val_split=data_config['val_split'],
            test_split=data_config['test_split'],
            seed=seed,
        )

        train_dataset = CombinedPotentialFieldDataset(
            samples=splits['train'],
            coord_range=coord_range,
            use_physical_coords=use_physical_coords,
            add_spacing_channels=add_spacing_channels,
            add_analytical_solution=add_analytical_solution,
        )

        val_dataset = CombinedPotentialFieldDataset(
            samples=splits['val'],
            coord_range=coord_range,
            use_physical_coords=use_physical_coords,
            add_spacing_channels=add_spacing_channels,
            add_analytical_solution=add_analytical_solution,
        )

        test_dataset = CombinedPotentialFieldDataset(
            samples=splits['test'],
            coord_range=coord_range,
            use_physical_coords=use_physical_coords,
            add_spacing_channels=add_spacing_channels,
            add_analytical_solution=add_analytical_solution,
        )
    else:
        # Use single directory (original behavior)
        train_indices, val_indices, test_indices = create_data_splits(
            data_dir=data_config['data_dir'],
            train_split=data_config['train_split'],
            val_split=data_config['val_split'],
            test_split=data_config['test_split'],
            seed=seed,
        )

        train_dataset = PotentialFieldDataset(
            data_dir=data_config['data_dir'],
            sample_indices=train_indices,
            coord_range=coord_range,
            use_physical_coords=use_physical_coords,
            add_spacing_channels=add_spacing_channels,
            add_analytical_solution=add_analytical_solution,
        )

        val_dataset = PotentialFieldDataset(
            data_dir=data_config['data_dir'],
            sample_indices=val_indices,
            coord_range=coord_range,
            use_physical_coords=use_physical_coords,
            add_spacing_channels=add_spacing_channels,
            add_analytical_solution=add_analytical_solution,
        )

        test_dataset = PotentialFieldDataset(
            data_dir=data_config['data_dir'],
            sample_indices=test_indices,
            coord_range=coord_range,
            use_physical_coords=use_physical_coords,
            add_spacing_channels=add_spacing_channels,
            add_analytical_solution=add_analytical_solution,
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
