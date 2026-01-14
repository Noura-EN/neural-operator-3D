# INSTRUCTIONS: Hybrid SciML for Potential Field Prediction

## 1. Objective
Build a modular PyTorch framework to predict the total electrical potential ($\Phi_{total}$) in a 3D volume. The approach is hybrid:
- **Total Potential**: $\Phi_{total} = \Phi_{analytical} \cdot (1 + \Phi_{correction})$
- **$\Phi_{analytical}$**: Analytical point-source solution in an infinite, homogeneous domain.
- **$\Phi_{correction}$**: Neural Network (FNO or UNet) predicting a relative correction field to account for heterogeneity ($\sigma$) and domain boundaries/geometry.


## 2. Data & Input Augmentation
- **Grid Size**: base resolution is $192 \times 96 \times 96$ (Cubic voxel grid).
- **Input channels (5 total)**:
    1. **Conductivity ($\sigma$)**: 3D volume. Implementation MUST support optional $\log_{10}(\sigma + \epsilon)$ transformation via config.
    2. **Source ($f$)**: Voxelized point source (delta function).
    3. **Coordinates (X, Y, Z)**: 3D meshgrids normalized from $[-1, 1]$. The meshgrid should be generated using torch.meshgrid(..., indexing='ij') to ensure spatial alignment with the voxelized conductivity data. **Essential for resolution-independent FNO performance.**
- **Ground Truth**: FEM-calculated potential ($\Phi_{FEM}$).

## 3. Architecture & Resolution Independence
- **Modular Backbones**: Swap between `3D-UNet` and `3D-FNO`. The code must be structured to swap backbones easily.
- **Geometry Encoder**: Placeholder `GeometryEncoder(nn.Module)` in `models/geometry.py` to process the structural features of the conductivity map. It should process the $\sigma$ volume into a latent feature map before passing it to the backbone.
- **Resolution Strategy**:
    - **Training**: Support training on base resolution OR downsampled (e.g., $96 \times 48 \times 48$).
    - **Inference**: FNO must accept high-res coordinate grids to evaluate the learned operator at new spatial points (Zero-Shot Super-Resolution).
    The codebase must support **Zero-Shot Super-Resolution** for the Neural Operator backbone.

### 4. Analytical Solver (`utils/analytical.py`)
Implement the infinite-domain potential function:
$$\Phi_{analytical}(\mathbf{r}) = \frac{I}{4\pi \sigma_{ref} \sqrt{|\mathbf{r} - \mathbf{r}_s|^2 + \epsilon^2}}$$

- **$\sigma_{ref}$**: The reference conductivity. Calculate this as the mean conductivity of the conductive tissues only (ignore "Air" voxels where $\sigma \approx 0$).
- **Source Centroid**: $\mathbf{r}_s$ should be determined by calculating the center of mass of the non-zero voxels in the source input $f$.
- **Singularity Softening**: Use a small $\epsilon$ (e.g., $0.1 \times$ voxel size) to prevent numerical overflow at the source center while maintaining a sharp peak for the neural operator to correct.


## 5. Physics-Informed Training
The total loss is a weighted sum: $\mathcal{L} = \lambda_1 \mathcal{L}_{MSE} + \lambda_2 \mathcal{L}_{PDE} + \lambda_3 \mathcal{L}_{Charge}$

1. **Data Loss ($\mathcal{L}_{MSE}$)**: 
   - Standard MSE between $\Phi_{total}$ and $\Phi_{FEM}$.
   - **Singularity Mask**: Mask a 3-voxel radius sphere around the source location $\mathbf{r}_s$ during loss calculation. This ensures the model focuses on the global field rather than trying to over-fit the analytical peak.

2. **PDE Loss ($\mathcal{L}_{PDE}$)**: 
   - Voxel-wise residual: $\nabla \cdot (\sigma \nabla \Phi_{total}) - f = 0$.
   - **Finite Difference**: Use a 7-point central difference stencil for the divergence of the gradient ($\sigma \nabla \Phi$).
   - **Grid Scaling**: The step size $h$ must be calculated dynamically based on the current grid resolution ($h = \frac{1}{N-1}$).
   - **Stability Mask**: Only calculate the PDE residual for voxels where $\sigma > \text{threshold}$. 

3. **Conservation of Charge Loss ($\mathcal{L}_{Charge}$)**: 
   - Instead of a local flux loss, implement a global conservation constraint.
   - Calculate the total divergence of the predicted current density: $D_{total} = \sum [\nabla \cdot (\sigma \nabla \Phi_{total})]$.
   - Calculate the total injected current: $S_{total} = \sum f$.
   - **Loss Component**: $\mathcal{L}_{Charge} = |D_{total} - S_{total}|^2$. 
   - This ensures the model respects the integral form of the Poisson equation across the entire volume, which is crucial for physical consistency in tissue potential modeling.

## 6. Execution & Evaluation
- **Train/Val/Test**: Standard split with performance tracking.
- **Logging**: Use `Weights & Biases` or `Tensorboard`.
- **Visuals**: Generate and save 2D slices (XY, YZ planes) comparing Ground Truth, Prediction and Correction term, and Error maps every 10 epochs for both train and val samples. For testing, generate the same, including results for different resolutions
- **Resolution Invariance**: Ensure FNO can be tested on a different grid resolution if requested (e.g. train on lower resolution, evaluate on higher).

## 7. Execution & Evaluation
- **Logging**: Weights & Biases or Tensorboard.
- **Visuals**: Every 10 epochs, save Axial and Sagittal slices showing:
    - [Ground Truth] | [Predicted Total] | [Neural Correction] | [Error Map]
- **Testing**: Run both normal testing and a "Resolution Invariance Test" where a model trained at $96 \times 48 \times 48$ is evaluated at $192 \times 96 \times 96$.

## 8. Project Structure
- `/data`: Voxelized .npy or .h5 files.
- `/src/models`: `unet.py`, `fno.py`, `geometry.py`, `hybrid.py`
- `/src/utils`: `physics.py` (PDE/Flux loss), `analytical.py` (Solver), `data_utils.py`
- `main.py`: Entry point for training/testing.
- `config.yaml` All hyperparameters (learning rate, λ weights, paths, resolution switches) must be centrally managed in a config.yaml file

### Implementation Note for `src/models/hybrid.py`:
The `HybridWrapper` class is the central integration point. It must:
1. Accept the 5-channel input tensor.
2. Extract the Normalized Coordinates to compute the `analytical_potential` on-the-fly.
3. Pass the input to the chosen Backbone (FNO or UNet) to obtain the `relative_correction`.
4. Combine them: `total_potential = analytical_potential * (1 + relative_correction)`.
5. Ensure all components remain on the same device (CPU/GPU) and maintain gradient flow for the PDE loss.

### 9. Hybrid Execution & HPC Compatibility (PBSPro)

The codebase must be "environment-aware," capable of running on a local workstation (Windows/Mac/Linux) or a High-Performance Computing (HPC) cluster using the **PBSPro** workload manager.

### A. Environment Detection & Device Management
- **Hardware Agnostic**: Implement a detection utility that checks for the existence of `PBS_NODEFILE`. 
- **Local Mode**: If no cluster environment is detected, default to a single-device setup (CUDA if available, otherwise CPU).
- **HPC Mode (DDP)**: If on the cluster, initialize `torch.nn.parallel.DistributedDataParallel` (DDP). 
    - Parse the `$PBS_NODEFILE` to dynamically set `WORLD_SIZE` and `MASTER_ADDR` (using the first node in the list).
    - Assign `LOCAL_RANK` and `RANK` by mapping the process to its index in the `PBS_NODEFILE`.
    - Use the `nccl` backend for GPU-to-GPU communication and `gloo` as a fallback.



### B. High-Performance I/O & Mixed Precision
- **Memory Optimization**: Use `torch.cuda.amp.autocast` (Mixed Precision) to reduce VRAM footprint. This is essential for fitting the $192 \times 96 \times 96$ volumes into memory on shared cluster GPUs.
- **Efficient Loading**:
    - **Local**: Use standard DataLoaders with `num_workers=4`.
    - **HPC**: Use `DistributedSampler` to ensure unique data batches per GPU. Scale `num_workers` to match the cluster's CPU-per-node allocation. Use `pin_memory=True`.
- **Checkpoint Resilience**: Automatically save and resume from `latest_checkpoint.pt`. This is mandatory to handle PBSPro wall-time limits; the model must be able to resume training seamlessly if the job is re-submitted.
When running on HPC, implement a flag to allow loading data from a local scratch directory (e.g., /scratch/ or $TMPDIR) to minimize network latency during training.

## 10. Master-Process Responsibilities (Rank 0)

To prevent file corruption and redundant logging in a multi-GPU environment, the following must only be executed by the process with `RANK == 0`:
- **Experiment Tracking**: Initialization of Weights & Biases or Tensorboard.
- **Visualizations**: Generation and saving of 2D Axial/Sagittal slices and error maps.
- **Model Checkpointing**: Writing weights to the `checkpoints/` directory.

## 11. Project Integration & Launch Scripts

### A. Modular Backbone Wrapper
- The `HybridWrapper` in `src/models/hybrid.py` must handle the 5-channel input (Source, Conductivity, X, Y, Z) and sum/multiply the components regardless of the underlying backbone (FNO or UNet).
- Ensure that the analytical grid generation happens on the correct device (matching the neural network's device).

### B. PBSPro Submission Template
Include a `scripts/submit_job.pbs` file that:
1. Allocates resources (nodes, GPUs, memory).
2. Sets up the Python environment and CUDA paths.
3. Calculates `MASTER_ADDR` and `WORLD_SIZE` from the PBS environment.
4. Uses `torchrun` to launch the distributed training across all nodes.