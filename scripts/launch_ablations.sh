#!/bin/bash
# Launch ablation experiments in the background
# This script manages the queue and runs 2 experiments at a time

cd /home/noura/Documents/Projects/PhD/simulation/neural-operator-3D

# First, let's wait for currently running experiments to complete
echo "=== Waiting for current experiments to complete ==="
while true; do
    running=$(ps aux | grep "python3 scripts/main.py" | grep -v grep | wc -l)
    if [ "$running" -eq 0 ]; then
        echo "Current experiments completed!"
        break
    fi
    echo "$(date): $running experiments still running..."
    sleep 120
done

# After current experiments complete, copy/rename their results for standardized naming
echo "=== Mapping current experiment results to ablation names ==="

# baseline_combined -> fno_analytical_standard_seed42
BASELINE_DIR=$(ls -d experiments/baseline_combined_* 2>/dev/null | head -1)
if [ -n "$BASELINE_DIR" ] && [ -f "$BASELINE_DIR/checkpoints/best_model.pt" ]; then
    TARGET="experiments/fno_analytical_standard_seed42"
    if [ ! -d "$TARGET" ]; then
        echo "Linking $BASELINE_DIR -> $TARGET"
        ln -s "$(basename $BASELINE_DIR)" "$TARGET" 2>/dev/null || true
    fi
fi

# fno_geom_attn_lite -> fno_geom_attn_standard_seed42
GEOM_DIR=$(ls -d experiments/fno_geom_attn_lite_* 2>/dev/null | head -1)
if [ -n "$GEOM_DIR" ] && [ -f "$GEOM_DIR/checkpoints/best_model.pt" ]; then
    TARGET="experiments/fno_geom_attn_standard_seed42"
    if [ ! -d "$TARGET" ]; then
        echo "Linking $GEOM_DIR -> $TARGET"
        ln -s "$(basename $GEOM_DIR)" "$TARGET" 2>/dev/null || true
    fi
fi

# Run full ablation queue
echo "=== Starting ablation experiments ==="
python3 scripts/run_ablations.py --max-parallel 2 --skip-completed

echo "=== All ablation experiments completed ==="

# Run evaluation on all experiments
echo "=== Running evaluation on all experiments ==="
for exp_dir in experiments/*/; do
    if [ -f "${exp_dir}checkpoints/best_model.pt" ] && [ ! -f "${exp_dir}evaluation_results.json" ]; then
        echo "Evaluating: $exp_dir"
        python3 scripts/evaluate_ablations.py --experiment-dir "$exp_dir"
    fi
done

# Aggregate results
echo "=== Aggregating results ==="
python3 scripts/aggregate_ablation_results.py

echo "=== DONE ==="
