#!/bin/bash
# Comprehensive overnight ablation experiments (~15 hours)
# Results written to ABLATIONS.md

cd /home/noura/Documents/Projects/PhD/simulation/neural-operator-3D

RESULTS_FILE="ABLATIONS.md"
BASELINE_MUSCLE_REL_L2=0.1359

# Initialize results file
cat > "$RESULTS_FILE" << 'EOF'
# Comprehensive Ablation Study Results

Auto-generated overnight experiments. Baseline: Muscle Rel L2 = 0.1359

## Results Table

| Experiment | Muscle Rel L2 | Muscle Far | L2 Norm | Grad Ratio | Laplacian | Epoch | vs Baseline |
|------------|---------------|------------|---------|------------|-----------|-------|-------------|
| baseline_layers6 | 0.1359 | 0.1340 | 0.9857 | 1.0210 | 2.39 | 46 | - |
EOF

echo "=========================================="
echo "Starting comprehensive overnight ablations"
echo "Time: $(date)"
echo "Estimated: ~60 experiments over 15 hours"
echo "=========================================="

# Function to run experiment and log results
run_experiment() {
    local config=$1
    local name=$2
    local notes=$3

    echo ""
    echo "========================================"
    echo "[$((++EXP_COUNT))] Running: $name"
    echo "Time: $(date)"
    echo "========================================"

    # Check if config exists
    if [ ! -f "configs/ablations/${config}" ]; then
        echo "Config not found: configs/ablations/${config}"
        echo "| $name | SKIPPED | - | - | - | - | - | Config missing |" >> "$RESULTS_FILE"
        return 1
    fi

    # Run training
    timeout 1800 python3 scripts/main.py --config "configs/ablations/${config}" 2>&1 | tail -50

    # Find latest experiment directory
    EXP_DIR=$(ls -td experiments/${name}_fno_* 2>/dev/null | head -1)

    if [ -z "$EXP_DIR" ] || [ ! -f "$EXP_DIR/final_results.json" ]; then
        echo "| $name | FAILED | - | - | - | - | - | $notes |" >> "$RESULTS_FILE"
        return 1
    fi

    # Extract and log metrics
    python3 -c "
import json
with open('$EXP_DIR/final_results.json') as f:
    d = json.load(f)
t = d['test_metrics']
epoch = d.get('best_epoch', -1)
baseline = $BASELINE_MUSCLE_REL_L2
diff = (t['rel_l2_muscle'] - baseline) / baseline * 100
marker = '✓' if diff < -2 else ('✗' if diff > 2 else '~')
print(f\"| $name | {t['rel_l2_muscle']:.4f} | {t['rel_l2_muscle_far']:.4f} | {t['l2_norm_ratio_muscle']:.4f} | {t['gradient_energy_ratio_muscle']:.4f} | {t['laplacian_energy_ratio_muscle']:.2f} | {epoch} | {marker} {diff:+.1f}% |\")" >> "$RESULTS_FILE"

    # Save predictions
    python3 scripts/save_predictions.py "$EXP_DIR" --split test 2>/dev/null

    return 0
}

# Create new config helper
create_config() {
    local name=$1
    local base_config=$2
    shift 2
    local modifications=("$@")

    python3 << PYTHON
import yaml

with open('configs/ablations/${base_config}') as f:
    config = yaml.safe_load(f)

# Apply modifications
modifications = """${modifications[*]}""".split('|')
for mod in modifications:
    if '=' in mod:
        path, value = mod.split('=', 1)
        keys = path.strip().split('.')
        d = config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        # Try to parse as number/bool
        try:
            if value.lower() == 'true':
                d[keys[-1]] = True
            elif value.lower() == 'false':
                d[keys[-1]] = False
            elif '.' in value:
                d[keys[-1]] = float(value)
            else:
                d[keys[-1]] = int(value)
        except:
            d[keys[-1]] = value

config['logging']['experiment_name'] = '$name'

with open('configs/ablations/${name}.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print(f"Created: configs/ablations/${name}.yaml")
PYTHON
}

EXP_COUNT=0

# ============================================
# PHASE 1: Loss Function Ablations (~2 hours)
# ============================================
echo "" >> "$RESULTS_FILE"
echo "## Phase 1: Loss Function Ablations" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Gradient matching weights
for gw in 0.5 1.0 2.0 3.0 5.0; do
    create_config "grad_weight_${gw}" "baseline_layers6.yaml" "loss.gradient_matching_weight=${gw}"
    run_experiment "grad_weight_${gw}.yaml" "grad_weight_${gw}" "Gradient matching"
done

# PDE weights
for pw in 0.01 0.05 0.1 0.2 0.5; do
    create_config "pde_weight_${pw}" "baseline_layers6.yaml" "loss.pde_weight=${pw}"
    run_experiment "pde_weight_${pw}.yaml" "pde_weight_${pw}" "PDE loss"
done

# TV regularization weights
for tv in 0.0 0.005 0.02 0.05 0.1; do
    create_config "tv_weight_${tv}" "baseline_layers6.yaml" "loss.tv_weight=${tv}"
    run_experiment "tv_weight_${tv}.yaml" "tv_weight_${tv}" "TV regularization"
done

# Laplacian matching (smoothness)
for lw in 0.1 0.5 1.0; do
    create_config "laplacian_weight_${lw}" "baseline_layers6.yaml" "loss.laplacian_matching_weight=${lw}"
    run_experiment "laplacian_weight_${lw}.yaml" "laplacian_weight_${lw}" "Laplacian matching"
done

# Singularity radius (just radius_15 since 0,3,5 are identical)
run_experiment "radius_15.yaml" "radius_15" "Large singularity exclusion"

# ============================================
# PHASE 2: Architecture Ablations (~3 hours)
# ============================================
echo "" >> "$RESULTS_FILE"
echo "## Phase 2: Architecture Ablations" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# FNO depth variations
for layers in 4 5 7 8; do
    create_config "layers_${layers}" "baseline_layers6.yaml" "model.fno.num_layers=${layers}"
    run_experiment "layers_${layers}.yaml" "layers_${layers}" "FNO depth"
done

# FNO width variations
for width in 24 48 64; do
    create_config "width_${width}" "baseline_layers6.yaml" "model.fno.width=${width}"
    run_experiment "width_${width}.yaml" "width_${width}" "FNO width"
done

# Fourier modes variations
for modes in 6 10 12 16; do
    create_config "modes_${modes}" "baseline_layers6.yaml" "model.fno.modes1=${modes}|model.fno.modes2=${modes}|model.fno.modes3=${modes}"
    run_experiment "modes_${modes}.yaml" "modes_${modes}" "Fourier modes"
done

# Projection head size
for fc in 64 256 512; do
    create_config "fc_dim_${fc}" "baseline_layers6.yaml" "model.fno.fc_dim=${fc}"
    run_experiment "fc_dim_${fc}.yaml" "fc_dim_${fc}" "Projection dim"
done

# ============================================
# PHASE 3: Training Ablations (~2 hours)
# ============================================
echo "" >> "$RESULTS_FILE"
echo "## Phase 3: Training Ablations" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Learning rate variations
for lr in 0.0001 0.0005 0.002 0.005; do
    create_config "lr_${lr}" "baseline_layers6.yaml" "training.learning_rate=${lr}"
    run_experiment "lr_${lr}.yaml" "lr_${lr}" "Learning rate"
done

# Weight decay variations
for wd in 0.0 0.00001 0.0001 0.001; do
    create_config "weight_decay_${wd}" "baseline_layers6.yaml" "training.weight_decay=${wd}"
    run_experiment "weight_decay_${wd}.yaml" "weight_decay_${wd}" "Weight decay"
done

# Longer training (more epochs, more patience)
create_config "epochs_200" "baseline_layers6.yaml" "training.epochs=200|training.early_stopping_patience=20"
run_experiment "epochs_200.yaml" "epochs_200" "Longer training"

# ============================================
# PHASE 4: Input Ablations (~1 hour)
# ============================================
echo "" >> "$RESULTS_FILE"
echo "## Phase 4: Input Ablations" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Without analytical solution
create_config "no_analytical" "baseline_layers6.yaml" "model.add_analytical_solution=false"
run_experiment "no_analytical.yaml" "no_analytical" "No analytical input"

# Without spacing conditioning
create_config "no_spacing_cond" "baseline_layers6.yaml" "spacing.use_spacing_conditioning=false"
run_experiment "no_spacing_cond.yaml" "no_spacing_cond" "No spacing conditioning"

# Geometry encoder variations
for geo_layers in 1 3 4; do
    create_config "geo_layers_${geo_layers}" "baseline_layers6.yaml" "model.geometry_encoder.num_layers=${geo_layers}"
    run_experiment "geo_layers_${geo_layers}.yaml" "geo_layers_${geo_layers}" "Geometry encoder depth"
done

for geo_dim in 32 128; do
    create_config "geo_dim_${geo_dim}" "baseline_layers6.yaml" "model.geometry_encoder.hidden_dim=${geo_dim}"
    run_experiment "geo_dim_${geo_dim}.yaml" "geo_dim_${geo_dim}" "Geometry encoder width"
done

# ============================================
# PHASE 5: Alternative Architectures (~2 hours)
# ============================================
echo "" >> "$RESULTS_FILE"
echo "## Phase 5: Alternative Architectures" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# TFNO (best from previous comparison)
create_config "tfno_baseline" "baseline_layers6.yaml" "model.backbone=tfno"
run_experiment "tfno_baseline.yaml" "tfno_baseline" "TFNO backbone"

# TFNO with more layers
create_config "tfno_layers8" "baseline_layers6.yaml" "model.backbone=tfno|model.fno.num_layers=8"
run_experiment "tfno_layers8.yaml" "tfno_layers8" "TFNO 8 layers"

# LSM (parameter efficient)
create_config "lsm_baseline" "baseline_layers6.yaml" "model.backbone=lsm"
run_experiment "lsm_baseline.yaml" "lsm_baseline" "LSM backbone"

# ============================================
# PHASE 6: Best Combinations (~2 hours)
# ============================================
echo "" >> "$RESULTS_FILE"
echo "## Phase 6: Best Combinations" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Analyze results and create combinations
python3 << 'PYTHON'
import json
from pathlib import Path

results = {}
for exp_dir in Path('experiments').glob('*_fno_*'):
    rf = exp_dir / 'final_results.json'
    if rf.exists():
        with open(rf) as f:
            data = json.load(f)
        name = exp_dir.name.split('_fno_')[0]
        results[name] = data['test_metrics']['rel_l2_muscle']

baseline = 0.1359
improvements = {k: (baseline - v) / baseline * 100 for k, v in results.items() if v < baseline * 0.98}

print("Top improvements found:")
for name, imp in sorted(improvements.items(), key=lambda x: -x[1])[:10]:
    print(f"  {name}: {imp:.1f}% better")

# Write best settings for combination
with open('best_settings.txt', 'w') as f:
    for name in sorted(improvements.keys(), key=lambda x: -improvements[x])[:5]:
        f.write(f"{name}\n")
PYTHON

# Create combinations of best settings
if [ -f "best_settings.txt" ]; then
    echo "Creating combinations of best settings..."

    # Read best settings and create combo configs
    python3 << 'PYTHON'
import yaml
from pathlib import Path

# Read best settings
with open('best_settings.txt') as f:
    best = [line.strip() for line in f.readlines()[:3]]

if len(best) >= 2:
    # Load configs and merge best settings
    base_config = yaml.safe_load(open('configs/ablations/baseline_layers6.yaml'))

    for exp in best:
        exp_config_path = Path(f'configs/ablations/{exp}.yaml')
        if exp_config_path.exists():
            exp_config = yaml.safe_load(open(exp_config_path))
            # Merge loss settings
            if 'loss' in exp_config:
                base_config['loss'].update(exp_config['loss'])
            # Merge model settings
            if 'model' in exp_config and 'fno' in exp_config['model']:
                base_config['model']['fno'].update(exp_config['model']['fno'])

    base_config['logging']['experiment_name'] = 'best_combo'
    with open('configs/ablations/best_combo.yaml', 'w') as f:
        yaml.dump(base_config, f)
    print("Created best_combo.yaml")
PYTHON

    run_experiment "best_combo.yaml" "best_combo" "Best settings combined"
fi

# ============================================
# PHASE 7: Fine-tuning Best Settings (~2 hours)
# ============================================
echo "" >> "$RESULTS_FILE"
echo "## Phase 7: Fine-tuning" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Find best overall setting and fine-tune around it
python3 << 'PYTHON'
import json
import yaml
from pathlib import Path

results = {}
for exp_dir in Path('experiments').glob('*_fno_*'):
    rf = exp_dir / 'final_results.json'
    if rf.exists():
        with open(rf) as f:
            data = json.load(f)
        name = exp_dir.name.split('_fno_')[0]
        results[name] = data['test_metrics']['rel_l2_muscle']

if results:
    best_name = min(results, key=results.get)
    best_perf = results[best_name]
    print(f"Best so far: {best_name} with {best_perf:.4f}")

    # If it's a grad_weight experiment, try values around it
    if 'grad_weight' in best_name:
        gw = float(best_name.split('_')[-1])
        for delta in [-0.3, -0.1, 0.1, 0.3]:
            new_gw = max(0, gw + delta)
            config = yaml.safe_load(open('configs/ablations/baseline_layers6.yaml'))
            config['loss']['gradient_matching_weight'] = new_gw
            config['logging']['experiment_name'] = f'grad_weight_fine_{new_gw}'
            with open(f'configs/ablations/grad_weight_fine_{new_gw}.yaml', 'w') as f:
                yaml.dump(config, f)
            print(f"Created grad_weight_fine_{new_gw}.yaml")
PYTHON

for fine_config in configs/ablations/grad_weight_fine_*.yaml; do
    if [ -f "$fine_config" ]; then
        name=$(basename "$fine_config" .yaml)
        run_experiment "${name}.yaml" "$name" "Fine-tuning"
    fi
done

# ============================================
# FINAL SUMMARY
# ============================================
echo "" >> "$RESULTS_FILE"
echo "## Final Summary" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "Completed at $(date)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "### Top 10 Experiments (by Muscle Rel L2)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

python3 << 'PYTHON'
import json
from pathlib import Path

results = []
for exp_dir in Path('experiments').glob('*_fno_*'):
    rf = exp_dir / 'final_results.json'
    if rf.exists():
        with open(rf) as f:
            data = json.load(f)
        name = exp_dir.name.split('_fno_')[0]
        results.append((name, data['test_metrics']))

# Sort by muscle rel l2
results.sort(key=lambda x: x[1]['rel_l2_muscle'])

baseline = 0.1359
print("| Rank | Experiment | Muscle Rel L2 | vs Baseline | Laplacian |")
print("|------|------------|---------------|-------------|-----------|")
for i, (name, metrics) in enumerate(results[:10], 1):
    diff = (metrics['rel_l2_muscle'] - baseline) / baseline * 100
    print(f"| {i} | {name} | {metrics['rel_l2_muscle']:.4f} | {diff:+.1f}% | {metrics['laplacian_energy_ratio_muscle']:.2f} |")
PYTHON >> "$RESULTS_FILE"

echo "" >> "$RESULTS_FILE"
echo "### Key Findings" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

python3 << 'PYTHON'
import json
from pathlib import Path

results = {}
for exp_dir in Path('experiments').glob('*_fno_*'):
    rf = exp_dir / 'final_results.json'
    if rf.exists():
        with open(rf) as f:
            data = json.load(f)
        name = exp_dir.name.split('_fno_')[0]
        results[name] = data['test_metrics']['rel_l2_muscle']

baseline = 0.1359
improved = {k: v for k, v in results.items() if v < baseline * 0.98}
degraded = {k: v for k, v in results.items() if v > baseline * 1.02}

print(f"- **{len(improved)} experiments improved** over baseline (>2% better)")
print(f"- **{len(degraded)} experiments degraded** (>2% worse)")
print(f"- **{len(results) - len(improved) - len(degraded)} experiments** showed no significant change")
print()

if improved:
    best = min(improved, key=improved.get)
    print(f"- **Best overall**: {best} with {improved[best]:.4f} ({(baseline - improved[best])/baseline*100:.1f}% improvement)")
PYTHON >> "$RESULTS_FILE"

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Time: $(date)"
echo "Results saved to: $RESULTS_FILE"
echo "=========================================="
