#!/usr/bin/env python3
"""Monitor ablation experiments and collect results."""
import os
import json
import glob
from datetime import datetime

def get_muscle_metrics(results_file):
    """Extract muscle-region metrics from results file."""
    try:
        with open(results_file) as f:
            data = json.load(f)
        test = data.get('test_metrics', {})
        return {
            'rel_l2_muscle': test.get('rel_l2_muscle', float('nan')),
            'rel_l2_muscle_far': test.get('rel_l2_muscle_far', float('nan')),
            'l2_norm_ratio_muscle': test.get('l2_norm_ratio_muscle', float('nan')),
            'gradient_energy_ratio_muscle': test.get('gradient_energy_ratio_muscle', float('nan')),
            'laplacian_energy_ratio_muscle': test.get('laplacian_energy_ratio_muscle', float('nan')),
            'epochs': data.get('total_epochs', 0),
        }
    except:
        return None

def main():
    print(f"\n{'='*80}")
    print(f"MUSCLE REGION ABLATION RESULTS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Find all experiment directories
    exp_dirs = sorted(glob.glob('experiments/*_fno_*'))
    
    # Categorize experiments
    radius_exps = {}
    grad_exps = {}
    pde_exps = {}
    other_exps = {}
    
    for exp_dir in exp_dirs:
        name = os.path.basename(exp_dir)
        results_file = os.path.join(exp_dir, 'final_results.json')
        ckpt_count = len(glob.glob(os.path.join(exp_dir, 'checkpoints', '*.pt')))
        
        if os.path.exists(results_file):
            metrics = get_muscle_metrics(results_file)
            status = 'DONE'
        else:
            metrics = None
            status = f'{ckpt_count} ckpts'
        
        if name.startswith('radius_'):
            radius = name.split('_')[1]
            radius_exps[radius] = (status, metrics)
        elif name.startswith('grad_weight_'):
            weight = name.split('_')[2]
            grad_exps[weight] = (status, metrics)
        elif name.startswith('pde_weight_'):
            weight = name.split('_')[2]
            pde_exps[weight] = (status, metrics)
        elif 'baseline' in name or 'layers6' in name:
            other_exps[name] = (status, metrics)
    
    # Print radius ablation results
    print("\n## RADIUS ABLATION (singularity exclusion)")
    print(f"{'Radius':<8} {'Status':<12} {'Muscle Rel L2':<14} {'Muscle Far':<12} {'Norm Ratio':<12} {'Grad Ratio':<12}")
    print("-" * 80)
    for radius in sorted(radius_exps.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        status, metrics = radius_exps[radius]
        if metrics:
            print(f"{radius:<8} {status:<12} {metrics['rel_l2_muscle']:<14.4f} {metrics['rel_l2_muscle_far']:<12.4f} {metrics['l2_norm_ratio_muscle']:<12.4f} {metrics['gradient_energy_ratio_muscle']:<12.4f}")
        else:
            print(f"{radius:<8} {status:<12} {'...':<14} {'...':<12} {'...':<12} {'...':<12}")
    
    # Print gradient weight results
    if grad_exps:
        print("\n## GRADIENT WEIGHT ABLATION")
        print(f"{'Weight':<8} {'Status':<12} {'Muscle Rel L2':<14} {'Muscle Far':<12} {'Norm Ratio':<12} {'Grad Ratio':<12}")
        print("-" * 80)
        for weight in sorted(grad_exps.keys(), key=lambda x: float(x) if x.replace('.','').isdigit() else 999):
            status, metrics = grad_exps[weight]
            if metrics:
                print(f"{weight:<8} {status:<12} {metrics['rel_l2_muscle']:<14.4f} {metrics['rel_l2_muscle_far']:<12.4f} {metrics['l2_norm_ratio_muscle']:<12.4f} {metrics['gradient_energy_ratio_muscle']:<12.4f}")
            else:
                print(f"{weight:<8} {status:<12} {'...':<14} {'...':<12} {'...':<12} {'...':<12}")
    
    # Print PDE weight results  
    if pde_exps:
        print("\n## PDE LOSS WEIGHT ABLATION")
        print(f"{'Weight':<8} {'Status':<12} {'Muscle Rel L2':<14} {'Muscle Far':<12} {'Norm Ratio':<12} {'Grad Ratio':<12}")
        print("-" * 80)
        for weight in sorted(pde_exps.keys(), key=lambda x: float(x) if x.replace('.','').isdigit() else 999):
            status, metrics = pde_exps[weight]
            if metrics:
                print(f"{weight:<8} {status:<12} {metrics['rel_l2_muscle']:<14.4f} {metrics['rel_l2_muscle_far']:<12.4f} {metrics['l2_norm_ratio_muscle']:<12.4f} {metrics['gradient_energy_ratio_muscle']:<12.4f}")
            else:
                print(f"{weight:<8} {status:<12} {'...':<14} {'...':<12} {'...':<12} {'...':<12}")
    
    # Print baseline comparison
    if other_exps:
        print("\n## BASELINE COMPARISON")
        for name, (status, metrics) in other_exps.items():
            if metrics:
                print(f"{name}: Muscle Rel L2 = {metrics['rel_l2_muscle']:.4f}, Far = {metrics['rel_l2_muscle_far']:.4f}")

if __name__ == '__main__':
    main()
