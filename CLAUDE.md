## Goal
3D neural operator for predicting electrical potential fields in muscle tissue simulations.
Governing PDE: -div(sigma * grad(phi)) = f with Neumann zero-flux BCs.

See `RESULTS.md` for full experimental results and conclusions.

## Dataset
- 500 low-res samples (48x48x96) in `data/`
- 500 high-res samples (96x96x192) in `data/voxel_96_96_192/`
- 400 downsampled high-res in `data/downsampled_highres/`
- 50 high-res training symlinks in `data/highres_training_samples/`
- 50 high-res test samples in `data/highres_test_samples/`

## Best Configuration
- **Architecture**: FNO (layers=6, modes=8, width=32)
- **Input**: Analytical monopole solution + conductivity + source + coordinates
- **Spacing**: Additive conditioning via MLP
- **Loss**: MSE + TV (weight=0.01), 3-voxel singularity mask
- **Mixed-res training**: Include 50 high-res samples for super-resolution

## Key Results
- **Base-res (FNO)**: Rel L2 = 0.152 +/- 0.020 (standard), 0.144 +/- 0.016 (mixed)
- **Super-res (FNO mixed)**: Rel L2 = 0.128 +/- 0.023
- **Ensemble (10 seeds)**: Rel L2 = 0.081 at high-res
- **MUAP correlation**: 0.993 (low-res), 0.937 (high-res)

## Available Backbones
- `fno` - Fourier Neural Operator (12.7M params) **[best]**
- `tfno` - Factorized FNO with instance normalization (12.7M params)
- `fno_geom_attn_lite` - FNO with geometry cross-attention (12.7M params)
- `unet` - 3D U-Net baseline (23.7M params)

## Running Experiments
```bash
# Single experiment
python scripts/main.py --config configs/fno_standard.yaml --experiment-name my_exp --seed 42

# Multi-seed study (10 seeds x 7 configs)
python scripts/run_ablations.py --max-parallel 2

# MUAP evaluation
python scripts/evaluate_muap_accuracy.py
```

## Hard Constraints
- max_epochs=100, early stopping patience=10
- Train/val/test: 70/15/15 split, seed=42
- Each experiment in its own `experiments/` subdirectory

## Metrics
- **Rel L2**: ||pred - target||_2 / ||target||_2
- **L2 Norm Ratio**: ||pred||_2 / ||target||_2 (1.0 = perfect scale)
- **Grad Energy Ratio**: mean(||grad pred||) / mean(||grad target||)
