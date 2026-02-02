#!/bin/bash
# Full queue: wait for radius_0, save predictions, then run remaining experiments

cd /home/noura/Documents/Projects/PhD/simulation/neural-operator-3D

echo "Waiting for radius_0 to complete..."

# Wait for current radius_0 to finish
while pgrep -f "main.py.*radius_0" > /dev/null; do
    sleep 30
    echo "  Still running... $(date '+%H:%M:%S')"
done

echo "radius_0 completed at $(date)"

# Save predictions for radius_0
EXP_DIR=$(ls -td experiments/radius_0_fno_* 2>/dev/null | head -1)
if [ -n "$EXP_DIR" ] && [ -f "$EXP_DIR/checkpoints/best_model.pt" ]; then
    echo "Saving predictions for radius_0..."
    python3 scripts/save_predictions.py "$EXP_DIR" --split test
fi

# Run remaining experiments
bash scripts/run_all_ablations.sh
