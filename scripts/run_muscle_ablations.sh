#!/bin/bash
cd /home/noura/Documents/Projects/PhD/simulation/neural-operator-3D

echo "Starting muscle region ablation experiments..."
echo "=============================================="

# Radius ablation (most important - find optimal radius)
for radius in 0 1 2 3 5 8 10; do
    echo ""
    echo "=== Running radius_${radius} ==="
    python3 scripts/main.py --config configs/ablations/radius_${radius}.yaml 2>&1 | tail -20
done

# Gradient weight ablation  
for grad in 1.0 2.0 3.0; do
    echo ""
    echo "=== Running grad_weight_${grad} ==="
    python3 scripts/main.py --config configs/ablations/grad_weight_${grad}.yaml 2>&1 | tail -20
done

# PDE loss ablation
for pde in 0.01 0.1 0.5; do
    echo ""
    echo "=== Running pde_weight_${pde} ==="
    python3 scripts/main.py --config configs/ablations/pde_weight_${pde}.yaml 2>&1 | tail -20
done

echo ""
echo "=============================================="
echo "All experiments completed!"
