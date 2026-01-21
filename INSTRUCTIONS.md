# PROMPT: Hybrid SciML for 3D Potential Field Prediction

### TL;DR
- Build a config-driven, modular PyTorch framework that learns the operator mapping \((\sigma, f, X, Y, Z, \Delta x, \Delta y, \Delta z) \rightarrow \Phi_{\text{total}}\) on a 3D domain, supports zero-shot super-resolution, and includes training/eval scripts, logging, masking losses, and tests.
- Prioritize clarity, modularity, and reproducibility; avoid hard-coding.
- Deliver runnable training (`scripts/main.py`), zero-shot eval (`scripts/eval_resolution.py`), models (`src/models/`), data pipeline (`src/data/`), utilities (`src/utils/`), configs (`configs/config.yaml`), and tests (`tests/`).
- Validate both UNet and FNO backbones via iterative runs on the data provided until they run without errors

## 1. Role & Context
You are an expert Senior Research Engineer specializing in Scientific Machine Learning (SciML) and PyTorch. Your task is to implement a **modular, operator-learning framework** to predict the **total electrical potential** \(\Phi_{\text{total}}\) in a **normalized 3D cubic volume** containing a **cylindrical-like geometry**.

The underlying physics is governed by a **Poisson-type problem** with **Neumann boundary conditions**:
\[
- \nabla \cdot (\sigma \nabla \Phi) = f
\]
with **zero-flux** at the boundary between skin and air voxels. The core requirement is **resolution independence / zero-shot super-resolution** via operator learning (e.g., Fourier Neural Operator, FNO): train at base resolution and generalize to higher resolutions **without retraining**.

## 2. Mathematical & Physical Specifications
- **Domain**: 3D voxel grid (base resolution: \(48 \times 48 \times 96\)) in a normalized coordinate cube \([-1, 1]^3\).
- **Physics**: Potential field governed by conductivity \(\sigma\).
  - **Isotropic regions**: scalar values (Fat, Bone, Skin, Air).
  - **Anisotropic (Muscle)**: \(3 \times 3\) diagonal tensor, \(\sigma_{muscle} = \text{diag}([0.2455, 0.2455, 1.2275])\) (ratio 1:1:5).
- **Boundary Conditions**: Neumann zero-flux (\(\nabla \Phi \cdot \mathbf{n} = 0\)) at domain boundary between skin and air.
- **Coordinates**: Normalized meshgrids \((X, Y, Z) \in [-1, 1]^3\) with `indexing='ij'`; provided as channels to enable zero-shot super-resolution.
- **Scale conditioning**: Physical voxel spacing \((\Delta x, \Delta y, \Delta z)\) must be explicitly injected (concatenate to latent or equivalent) to disambiguate scale-dependent gradients.

## 3. Data & Input Pipeline
Implement data handling under `src/data/`.

### Channels and shapes
- **Conductivity \(\sigma\)**: voxel map; muscle uses 3 channels for diagonal tensor (or consistent encoding). Shape `(C_sigma, D, H, W)`.
- **Source \(f\)**: scalar field; shape `(1, D, H, W)`.
- **Coordinates \(X, Y, Z\)**: normalized meshgrids; shape `(3, D, H, W)`.
- **Voxel spacing**: vector `(3,)` for \(\Delta x, \Delta y, \Delta z\); later concatenated/conditioned in-network.
- **Ground truth**: \(\Phi_{FEM}\) with shape `(1, D, H, W)` used as the supervised training target.

### Loader requirements (`src/data/loader.py`)
- Load \(\sigma, f, X, Y, Z, \text{spacing}, \Phi_{FEM}\).
- Support both training resolution (e.g., \(48 \times 48 \times 96\)) and higher-resolution eval (e.g., \(96 \times 96 \times 192\)).
- Device-agnostic moves (cuda/mps/cpu).
- Provide deterministic seeding hooks from config.

## 4. Model Architecture: Modular backbone strategy
Implement all model-related code in `src/models/`.

- **Pattern**: Configurable backbone selected via wrapper:
  - Shared **Geometry Encoder** to process conductivity.
  - Backbones: **3D-UNet** and **3D-FNO**.

- **Geometry Encoder (`geometry.py`)**
  - `GeometryEncoder(nn.Module)`; 3D conv-based volumetric encoder.
  - Input: \(\sigma\) channels (including anisotropic components).
  - Output: high-dimensional feature volume aligned to the grid.
  - Reusable across backbones.

- **Backbone 1: 3D-UNet (`unet.py`)**
  - Inputs: geometry features, \(f\), \(X, Y, Z\), spacing (concatenate or inject).
  - Output: \(\Phi_{\text{pred}}\) on the same grid.
  - Serves as baseline to validate pipeline.

- **Backbone 2: 3D-FNO (`fno.py`)**
  - Use `torch.fft.rfftn` / `torch.fft.irfftn` for spectral convolutions (memory-efficient, real-valued symmetry).
  - Inputs: geometry features, \(f\), \(X, Y, Z\), explicit spacing conditioning.
  - Output: \(\Phi_{\text{pred}}\) on same grid.
  - Must train at base resolution and infer at higher resolutions without retraining (zero-shot super-resolution).

- **Model Wrapper (`wrapper.py`)**
  - Accept config specifying `"unet"` or `"fno"`.
  - Instantiate `GeometryEncoder` + chosen backbone.
  - Expose `forward(self, sigma, f, coords, spacing, **kwargs) -> Phi_pred`.
  - Handle internal conditioning concatenation (including spacing).

## 5. Masking & Physics-Informed Training
Implement losses/masking in `src/utils/masking.py` and `src/utils/metrics.py`.

- **Weighted Masked MSE**:
  \[
  \mathcal{L} = \lambda \cdot \text{MSE}(\Phi_{pred} \cdot M_{combined}, \Phi_{FEM} \cdot M_{combined})
  \]
    - **Masks**:
    - Muscle mask \(M_m\): binary where \(\sigma\) corresponds to muscle (match anisotropic tensor).
    - Singularity mask \(M_s\): binary sphere radius \(R=3\) voxels around source peak \(\mathbf{r}_s\) (infer from peak of \(f\)).
    - Combined \(M_{combined} = M_m \cap \neg M_s\).
- **Gradient consistency loss** \(\mathcal{L}_{grad}\):
  - Compute \(\mathbf{E} = -\nabla \Phi\) with `torch.gradient`, respecting spacing.
  - MSE between predicted and ground-truth gradients, weighted \(\lambda_{grad} = 0.1\).

## 6. Execution, Logging & Evaluation
- **Training entry (`scripts/main.py`)**
  - Config-driven CLI.
  - Load hyperparameters from `configs/config.yaml`.
  - Build datasets/dataloaders, model via wrapper, optimizer/scheduler, losses.
  - Run train/val loops with logging.

- **Configuration (`configs/config.yaml`)**
  - Hyperparameters: lr, batch size, epochs, FNO modes, model type (unet/fno), loss weights, logging/validation frequency, experiment name, seeds, device preference.

- **Logging**
  - Use W&B or TensorBoard (pick one, be consistent).
  - Log train/val losses, gradient/parameter norms (periodically), and full hyperparameter config snapshot.

- **Checkpoints**
    - Save checkpoints in `experiments/{experiment_name}/...`

- **Validation visuals**
  - Every 10 epochs, log 3-panel figure (axial/sagittal slice):
    - Masked ground-truth potentials
    - Masked predictions
    - Log-absolute error
  - Implement plotting in `src/utils/visualization.py`.

- **Zero-shot super-resolution eval (`scripts/eval_resolution.py`)**
  - Load model trained at \(48 \times 48 \times 96\).
  - Infer at higher resolution (e.g., \(96 \times 96 \times 192\)) over same \([-1, 1]^3\) domain.
  - Log predictions, error maps, and quantitative metrics (MSE, relative error) on the high-res grid.

## 7. Project Structure & Standards
```text
/
├── configs
│   └── config.yaml          # hyperparams (lr, batch_size, FNO_modes, loss weights, etc)
├── data/
├── experiments/
├── scripts
│   ├── main.py              # trainer entry point (config-driven)
│   └── eval_resolution.py   # zero-shot super-resolution evaluation
├── src/
│   ├── models/              # geometry.py, unet.py, fno.py, wrapper.py
│   ├── data/                # loader.py, transforms.py
│   └── utils/               # masking.py, metrics.py, visualization.py
└── tests/                   # unit tests for FNO spectral convolutions and key utilities
```

## 8. Implementation Constraints and Best Practices
- Memory efficiency: use `torch.fft.rfftn` for 3D FNO to exploit real-valued symmetries.
- Device agnostic:
  - Auto-select device: prefer cuda, else cpu (support mps if available).
  - Move tensors/models consistently.
- Reproducibility: set seeds for torch, numpy, random via config.
- Config-driven design:
  - Avoid hard-coded hyperparameters.
  - Control model choice, sizes, training schedule, loss weights, masks behavior via config/CLI.

## 9. What to prioritise
1. Data pipeline (loader.py, coords/spacing generation, FEM target loading).
2. Baseline UNet pipeline (GeometryEncoder, masking, losses, logging).
3. FNO backbone (rFFT-based, memory-efficient).
4. Zero-shot resolution evaluation script (eval_resolution.py).
5. Visualization, metrics, and unit tests.
6. Clarity, modularity, correctness; easy to plug new operator learners.
7. Run with example data to smoke-test workflow.

## 10. Definition of Done
- Config-driven training and eval scripts run end-to-end on sample data.
- For **both** backbones (`unet` and `fno`), iteratively run a **smoke test** on the 100 sample example data (train + val + logging). After each run, fix any failures/bugs and rerun until it completes end-to-end with **no runtime errors**.
- Models support base training and higher-res inference without code changes.
- Masked loss and gradient loss implemented and used in training.
- Logging produces losses, norms, configs, and periodic visuals.
- Tests cover FNO spectral ops and key utilities.

---

## Implementation Notes & Clarifications

### Data Format Clarifications
- **Data files**: `.npz` format with keys: `sigma`, `source`, `mask`, `u`, `spacing`, `source_point`
- **Conductivity tensor**: Stored as `(D, H, W, 6)` with 6 channels representing symmetric 3x3 tensor (3 diagonal + 3 off-diagonal)
- **Coordinate grids**: Generated on-the-fly from grid shape and `coord_range`, not stored in data files
- **Resolution**: The actual data resolution is `(D=96, H=48, W=48)`. Training uses native resolution; config resolution settings are for super-resolution evaluation.

### Implementation Decisions
- **Logging**: TensorBoard chosen (configured in `config.yaml` with `use_tensorboard: true`)
- **Device selection**: Automatic preference: CUDA > MPS > CPU
- **Spacing conditioning**: Implemented via `SpacingConditioner` module with learnable multiplicative modulation

### Smoke Test Results
Both UNet and FNO backbones completed 10-epoch smoke tests successfully on 100-sample dataset:

| Backbone | Parameters | Final Val Loss | Best Rel L2 |
|----------|-----------|----------------|-------------|
| UNet     | 23.7M     | 0.000004       | 0.728       |
| FNO      | 8.5M      | 0.000001       | 0.283       |

### Running the Framework

```bash
# Train with FNO (default)
python scripts/main.py --config configs/config.yaml

# Train with UNet (modify config.yaml: backbone: unet)
python scripts/main.py --config configs/config.yaml

# Zero-shot super-resolution evaluation
python scripts/eval_resolution.py --checkpoint experiments/.../checkpoints/best_model.pt --visualize
```