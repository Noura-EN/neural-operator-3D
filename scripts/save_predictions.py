#!/usr/bin/env python3
"""Save model predictions and ground truth for analysis."""

import argparse
import os
import torch
import numpy as np
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import get_dataloaders
from src.models.wrapper import PotentialFieldModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir', help='Path to experiment directory')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--output-dir', default=None, help='Output directory (default: experiment_dir/predictions)')
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)

    # Load config
    config_path = exp_dir / 'config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else exp_dir / 'predictions'
    output_dir.mkdir(exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    if args.split == 'train':
        data_loader = train_loader
    elif args.split == 'val':
        data_loader = val_loader
    else:
        data_loader = test_loader

    # Load model
    print("Loading model...")
    model_config = config.get('model', {})
    fno_config = model_config.get('fno', {})
    geo_config = model_config.get('geometry_encoder', {})

    model = PotentialFieldModel(
        backbone_type=model_config.get('backbone', 'fno'),
        geometry_hidden_dim=geo_config.get('hidden_dim', 64),
        geometry_num_layers=geo_config.get('num_layers', 2),
        fno_config={
            'modes1': fno_config.get('modes1', 8),
            'modes2': fno_config.get('modes2', 8),
            'modes3': fno_config.get('modes3', 8),
            'width': fno_config.get('width', 32),
            'num_layers': fno_config.get('num_layers', 4),
            'fc_dim': fno_config.get('fc_dim', 128),
        },
        add_analytical_solution=model_config.get('add_analytical_solution', False),
        use_spacing_conditioning=config.get('spacing', {}).get('use_spacing_conditioning', True),
    )

    # Load weights
    checkpoint_path = exp_dir / 'checkpoints' / 'best_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")

    # Run inference and save
    print(f"Running inference on {args.split} set...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            sigma = batch['sigma'].to(device)
            source = batch['source'].to(device)
            coords = batch['coords'].to(device)
            spacing = batch['spacing'].to(device)
            target = batch['target'].to(device)

            analytical = batch.get('analytical', None)
            if analytical is not None:
                analytical = analytical.to(device)

            # Run model
            pred = model(sigma, source, coords, spacing, analytical=analytical)

            # Save as NPZ
            output_file = output_dir / f'{args.split}_sample_{batch_idx:04d}.npz'
            np.savez(
                output_file,
                prediction=pred.cpu().numpy(),
                target=target.cpu().numpy(),
                sigma=sigma.cpu().numpy(),
                source=source.cpu().numpy(),
                spacing=spacing.cpu().numpy(),
            )

            if (batch_idx + 1) % 10 == 0:
                print(f"  Saved {batch_idx + 1} samples...")

    print(f"\nSaved {batch_idx + 1} samples to {output_dir}")
    print("Files contain: prediction, target, sigma, source, spacing")


if __name__ == '__main__':
    main()
