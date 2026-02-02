#!/bin/bash
# Run all ablation experiments sequentially
# Edit the EXPERIMENTS array below to change order or skip experiments

cd /home/noura/Documents/Projects/PhD/simulation/neural-operator-3D

# Experiments to run (in order) - EDIT THIS TO CHANGE ORDER
EXPERIMENTS=(
    # Radius ablation (0, 3, 5 done - all identical)
    "radius_8"
    # Gradient weight ablation
    "grad_weight_1.0"
    "grad_weight_2.0"
    "grad_weight_3.0"
    # PDE weight ablation
    "pde_weight_0.01"
    "pde_weight_0.1"
    "pde_weight_0.5"
)

echo "=============================================="
echo "Running ${#EXPERIMENTS[@]} ablation experiments"
echo "Edit this script to change order (before experiment starts)"
echo "=============================================="

for exp in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "======================================"
    echo "Starting: $exp"
    echo "Time: $(date)"
    echo "======================================"

    # Run training
    python3 scripts/main.py --config "configs/ablations/${exp}.yaml" 2>&1

    # Find the most recent experiment directory for this config
    EXP_DIR=$(ls -td experiments/${exp}_fno_* 2>/dev/null | head -1)

    if [ -n "$EXP_DIR" ] && [ -f "$EXP_DIR/checkpoints/best_model.pt" ]; then
        echo "Saving predictions for $exp..."
        python3 scripts/save_predictions.py "$EXP_DIR" --split test
        echo "Completed: $exp -> $EXP_DIR"
    else
        echo "WARNING: No checkpoint found for $exp"
    fi

    echo ""
done

echo "=============================================="
echo "All experiments completed at $(date)"
echo "=============================================="
