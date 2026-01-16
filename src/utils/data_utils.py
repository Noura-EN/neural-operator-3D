"""
Data loading utilities with HPC support.
Handles .npy and .h5 file formats for voxelized data.
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F


class PotentialFieldDataset(Dataset):
    """
    Dataset for loading potential field data.
    Expected format: .h5 or .npy files containing:
    - conductivity: (D, H, W) or (1, D, H, W)
    - source: (D, H, W) or (1, D, H, W)
    - potential: (D, H, W) or (1, D, H, W) - ground truth
    """
    
    def __init__(
        self,
        data_dir: str,
        file_list: list = None,
        resolution: tuple = None,
        log_conductivity: bool = False,
        conductivity_epsilon: float = 1e-6,
        coord_range: tuple = (-1.0, 1.0)
    ):
        """
        Args:
            data_dir: Directory containing data files
            file_list: List of file names to use (if None, uses all .h5/.npy files)
            resolution: Target resolution (D, H, W) for resampling
            log_conductivity: Whether to apply log10 transformation to conductivity
            conductivity_epsilon: Epsilon for log transformation
            coord_range: Range for coordinate normalization
        """
        self.data_dir = data_dir
        self.resolution = resolution
        self.log_conductivity = log_conductivity
        self.conductivity_epsilon = conductivity_epsilon
        self.coord_range = coord_range
        
        # Get file list
        if file_list is None:
            self.file_list = []
            for f in os.listdir(data_dir):
                if f.endswith('.h5') or f.endswith('.npy') or f.endswith('.npz'):
                    self.file_list.append(f)
        else:
            self.file_list = file_list
        
        self.file_list.sort()
    
    def __len__(self):
        return len(self.file_list)
    
    def _load_file(self, idx):
        """Load a single data file."""
        filepath = os.path.join(self.data_dir, self.file_list[idx])
        
        if filepath.endswith('.h5'):
            with h5py.File(filepath, 'r') as f:
                conductivity = np.array(f['conductivity'])
                source = np.array(f['source'])
                potential = np.array(f['potential'])
        elif filepath.endswith('.npz'):
            # Handle .npz format
            data = np.load(filepath, allow_pickle=True)
            
            # Map common key variations
            # Try 'sigma' or 'conductivity' for conductivity
            if 'sigma' in data:
                conductivity = data['sigma']
            elif 'conductivity' in data:
                conductivity = data['conductivity']
            else:
                raise KeyError(f"Could not find 'sigma' or 'conductivity' in {filepath}")
            
            # Try 'u' or 'potential' for potential
            if 'u' in data:
                potential = data['u']
            elif 'potential' in data:
                potential = data['potential']
            else:
                raise KeyError(f"Could not find 'u' or 'potential' in {filepath}")
            
            # Source should be present
            if 'source' not in data:
                raise KeyError(f"Could not find 'source' in {filepath}")
            source = data['source']
            
        else:  # .npy - assume it's a dictionary saved as .npz
            data = np.load(filepath, allow_pickle=True)
            if isinstance(data, np.lib.npyio.NpzFile):
                # Try to get conductivity/sigma
                if 'conductivity' in data:
                    conductivity = data['conductivity']
                elif 'sigma' in data:
                    conductivity = data['sigma']
                else:
                    raise KeyError(f"Could not find 'conductivity' or 'sigma' in {filepath}")
                
                # Try to get potential/u
                if 'potential' in data:
                    potential = data['potential']
                elif 'u' in data:
                    potential = data['u']
                else:
                    raise KeyError(f"Could not find 'potential' or 'u' in {filepath}")
                
                source = data['source']
            else:
                raise ValueError(f"Unsupported .npy format in {filepath}")
        
        # Handle multi-channel conductivity (take first channel or mean if needed)
        if conductivity.ndim == 4:
            # If last dimension is channels, take first channel or mean
            if conductivity.shape[-1] <= 10:  # Likely channel dimension
                conductivity = conductivity[..., 0]  # Take first channel
            else:
                conductivity = conductivity[0]  # Take first sample
        elif conductivity.ndim > 3:
            # Flatten extra dimensions
            while conductivity.ndim > 3:
                conductivity = conductivity[0]
        
        # Ensure 3D shape for all arrays
        if source.ndim == 4:
            source = source[0]
        if potential.ndim == 4:
            potential = potential[0]
        
        # Ensure all are 3D
        assert conductivity.ndim == 3, f"Conductivity should be 3D, got shape {conductivity.shape}"
        assert source.ndim == 3, f"Source should be 3D, got shape {source.shape}"
        assert potential.ndim == 3, f"Potential should be 3D, got shape {potential.shape}"
        
        return conductivity, source, potential
    
    def _resample(self, data: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Resample data to target resolution."""
        if data.shape == target_shape:
            return data
        
        # Convert to torch tensor for interpolation
        data_tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        
        # Resample using trilinear interpolation
        data_resampled = F.interpolate(
            data_tensor,
            size=target_shape,
            mode='trilinear',
            align_corners=False
        )
        
        return data_resampled.squeeze().numpy()
    
    def _generate_coordinates(self, shape: tuple) -> np.ndarray:
        """Generate normalized coordinate grids."""
        D, H, W = shape
        coord_min, coord_max = self.coord_range
        
        # Generate meshgrid
        d = np.linspace(coord_min, coord_max, D)
        h = np.linspace(coord_min, coord_max, H)
        w = np.linspace(coord_min, coord_max, W)
        
        dd, hh, ww = np.meshgrid(d, h, w, indexing='ij')
        
        # Stack into (3, D, H, W)
        coords = np.stack([dd, hh, ww], axis=0)
        
        return coords
    
    def __getitem__(self, idx):
        # Load data
        conductivity, source, potential = self._load_file(idx)
        
        # Resample if needed
        if self.resolution is not None:
            conductivity = self._resample(conductivity, self.resolution)
            source = self._resample(source, self.resolution)
            potential = self._resample(potential, self.resolution)
        
        # Generate coordinates
        coords = self._generate_coordinates(conductivity.shape)
        
        # Apply log transformation to conductivity if requested
        if self.log_conductivity:
            conductivity = np.log10(conductivity + self.conductivity_epsilon)
        
        # Convert to tensors
        conductivity = torch.from_numpy(conductivity).float().unsqueeze(0)  # (1, D, H, W)
        source = torch.from_numpy(source).float().unsqueeze(0)  # (1, D, H, W)
        potential = torch.from_numpy(potential).float().unsqueeze(0)  # (1, D, H, W)
        coords = torch.from_numpy(coords).float()  # (3, D, H, W)
        
        # Stack into 5-channel input: [source, conductivity, X, Y, Z]
        input_tensor = torch.cat([
            source,  # (1, D, H, W)
            conductivity,  # (1, D, H, W)
            coords  # (3, D, H, W)
        ], dim=0)  # (5, D, H, W)
        
        return {
            'input': input_tensor,  # (5, D, H, W)
            'potential': potential,  # (1, D, H, W)
            'conductivity': conductivity,  # (1, D, H, W)
            'source': source,  # (1, D, H, W)
            'coords': coords  # (3, D, H, W)
        }


def create_dataloaders(
    data_dir: str,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    batch_size: int = 2,
    num_workers: int = 4,
    pin_memory: bool = True,
    resolution: tuple = None,
    log_conductivity: bool = False,
    conductivity_epsilon: float = 1e-6,
    coord_range: tuple = (-1.0, 1.0),
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    use_scratch_dir: bool = False,
    scratch_dir: str = "/scratch"
):
    """
    Create train/val/test dataloaders.
    
    Args:
        data_dir: Base data directory
        train_split, val_split, test_split: Data split ratios
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        resolution: Target resolution (D, H, W)
        log_conductivity: Whether to log-transform conductivity
        conductivity_epsilon: Epsilon for log transformation
        coord_range: Coordinate normalization range
        use_distributed: Whether using distributed training
        rank: Process rank
        world_size: Total number of processes
        use_scratch_dir: Whether to use scratch directory on HPC
        scratch_dir: Scratch directory path
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Handle scratch directory on HPC
    if use_scratch_dir and os.path.exists(scratch_dir):
        # Copy or symlink data to scratch (simplified - in practice might want to copy)
        actual_data_dir = scratch_dir
        if not os.path.exists(actual_data_dir):
            actual_data_dir = data_dir
    else:
        actual_data_dir = data_dir
    
    # Create full dataset
    full_dataset = PotentialFieldDataset(
        data_dir=actual_data_dir,
        resolution=resolution,
        log_conductivity=log_conductivity,
        conductivity_epsilon=conductivity_epsilon,
        coord_range=coord_range
    )
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Handle edge case: if only 1 sample, use it for training
    if total_size == 1:
        train_size = 1
        val_size = 0
        test_size = 0
    # Handle edge case: if very few samples, ensure at least 1 in train
    elif train_size == 0 and total_size > 0:
        train_size = 1
        if val_size > 0:
            val_size -= 1
        if test_size > 0:
            test_size -= 1
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create samplers
    if use_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
        shuffle_train = True
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader
