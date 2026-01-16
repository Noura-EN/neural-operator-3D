# Hybrid SciML for Potential Field Prediction

A modular PyTorch framework for predicting electrical potential fields in 3D volumes using a hybrid approach that combines analytical solutions with neural network corrections.

## Overview

This framework implements a hybrid model that predicts the total electrical potential $\Phi_{total}$ as:

$$\Phi_{total} = \Phi_{analytical} \cdot (1 + \Phi_{correction})$$

where:
- **$\Phi_{analytical}$**: Analytical point-source solution in an infinite, homogeneous domain
- **$\Phi_{correction}$**: Neural network (FNO or UNet) predicting a relative correction field to account for heterogeneity and domain boundaries

## Features

- **Modular Architecture**: Easily swap between 3D-UNet and 3D-FNO backbones
- **Zero-Shot Super-Resolution**: FNO supports resolution-independent inference
- **Physics-Informed Training**: Combined MSE, PDE residual, and charge conservation losses
- **HPC Compatible**: Automatic detection and setup for PBSPro clusters with distributed training
- **Mixed Precision Training**: Automatic mixed precision (AMP) for memory efficiency
- **Checkpointing**: Automatic checkpoint saving and resuming for long training runs

## Installation

1. Clone the repository and navigate to the project directory:
```bash
cd simulation/neural-operator-3D
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
neural-operator-3D/
├── config.yaml              # Configuration file
├── main.py                  # Main training/testing script
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/                   # Data directory (place .h5 or .npy files here)
├── checkpoints/            # Model checkpoints
├── logs/                   # Training logs (TensorBoard)
├── visualizations/         # Generated visualizations
├── scripts/
│   └── submit_job.pbs     # PBSPro submission script
└── src/
    ├── models/
    │   ├── hybrid.py       # Hybrid wrapper combining analytical + neural
    │   ├── fno.py          # 3D Fourier Neural Operator
    │   ├── unet.py         # 3D U-Net
    │   └── geometry.py     # Geometry encoder
    └── utils/
        ├── analytical.py   # Analytical potential solver
        ├── physics.py      # Physics-informed loss functions
        ├── data_utils.py   # Data loading utilities
        └── hpc_utils.py    # HPC environment detection
```

## Data Format

The framework expects data files in `.h5` or `.npy` format containing:
- `conductivity`: 3D conductivity field (D, H, W) or (1, D, H, W)
- `source`: Voxelized point source (D, H, W) or (1, D, H, W)
- `potential`: Ground truth potential field (D, H, W) or (1, D, H, W)

For `.h5` files, use keys: `'conductivity'`, `'source'`, `'potential'`
For `.npy` files, use `.npz` format with the same keys.

## Configuration

All hyperparameters are configured in `config.yaml`. Key sections:

- **Data**: Data paths, batch size, train/val/test splits
- **Model**: Backbone selection (FNO/UNet), architecture parameters
- **Training**: Learning rate, loss weights, epochs, checkpointing
- **Grid**: Resolution settings, coordinate normalization
- **Physics**: Analytical solver parameters, PDE loss thresholds
- **HPC**: Distributed training settings

## Usage

### Local Training

```bash
python main.py --config config.yaml --mode train
```

### Testing

```bash
python main.py --config config.yaml --mode test --checkpoint checkpoints/best_checkpoint.pt
```

### HPC (PBSPro) Submission

1. Edit `scripts/submit_job.pbs` to set:
   - Resource requirements (nodes, GPUs, memory, walltime)
   - Python environment path
   - Working directory

2. Submit the job:
```bash
qsub scripts/submit_job.pbs
```

The script automatically:
- Detects PBSPro environment
- Parses nodefile for distributed setup
- Launches training with `torchrun`

## Key Components

### Analytical Solver

Implements the infinite-domain potential:
$$\Phi_{analytical}(\mathbf{r}) = \frac{I}{4\pi \sigma_{ref} \sqrt{|\mathbf{r} - \mathbf{r}_s|^2 + \epsilon^2}}$$

### Physics-Informed Losses

1. **MSE Loss**: Masked around source singularity (3-voxel radius)
2. **PDE Loss**: Residual of $\nabla \cdot (\sigma \nabla \Phi) - f = 0$ using 7-point stencil
3. **Charge Conservation**: Global constraint $|D_{total} - S_{total}|^2$

### Zero-Shot Super-Resolution

The FNO backbone operates in Fourier space, enabling evaluation at different resolutions than training. Set `test_resolutions` in config to test on multiple grid sizes.

## Visualization

Visualizations are automatically generated every `vis_freq` epochs (default: 10) showing:
- Ground Truth | Predicted Total | Neural Correction | Error Map

Slices are saved in `visualizations/` directory for both training and validation sets.

## Checkpointing

Checkpoints are automatically saved:
- `latest_checkpoint.pt`: Most recent checkpoint
- `best_checkpoint.pt`: Best validation loss checkpoint
- `checkpoint_epoch_N.pt`: Periodic checkpoints

Resume training by setting `resume_from` in config or using the latest checkpoint path.

## Distributed Training

The framework automatically detects HPC environments and sets up distributed training:
- Uses `PBS_NODEFILE` to determine node configuration
- Initializes DDP with `nccl` backend (GPU) or `gloo` (CPU)
- Only rank 0 process handles logging, visualization, and checkpointing

## Citation

If you use this framework, please cite the relevant papers for:
- Fourier Neural Operators (Li et al., 2020)
- Physics-Informed Neural Networks (Raissi et al., 2019)

## License

[Add your license here]

## Contact

[Add contact information here]
