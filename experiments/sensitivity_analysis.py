"""Sensitivity analysis for key hyperparameters"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_experiment import run_uapbr_experiment

def run_sensitivity_analysis(base_config):
    """Analyze sensitivity to key parameters"""
    
    # Parameters to vary
    param_ranges = {
        'lambda_cost': [0.1, 0.2, 0.3, 0.4, 0.5],
        'resolution': [16, 32, 64],
        'n_samples': [100, 300, 500, 1000],
        'ae_epochs': [5, 10, 20],
        'batch_size': [16, 32, 64]
    }
    
    results = {}
    
    for param_name, values in param_ranges.items():
        print(f"\n📊 Analyzing sensitivity to {param_name}...")
        param_results = []
        
        for val in values:
            config = base_config.copy()
            config[param_name] = val
            
            # Run experiment
            exp_results = run_uapbr_experiment(config)
            param_results.append({
                'value': val,
                'risk': exp_results['metrics']['empirical_risk'],
                'accuracy': exp_results['metrics']['accuracy_accepted'],
                'acceptance': exp_results['metrics']['acceptance_rate']
            })
        
        results[param_name] = param_results
    
    # Plot sensitivity
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (param_name, param_results) in enumerate(results.items()):
        if i >= 6:
            break
            
        vals = [r['value'] for r in param_results]
        risks = [r['risk'] for r in param_results]
        accs = [r['accuracy'] for r in param_results]
        
        ax = axes[i]
        ax.plot(vals, risks, 'o-', label='Risk', linewidth=2)
        ax.plot(vals, accs, 's-', label='Accuracy', linewidth=2)
        ax.set_xlabel(param_name)
        ax.set_ylabel('Value')
        ax.set_title(f'Sensitivity to {param_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sensitivity_analysis.pdf', bbox_inches='tight')
    plt.show()
    
    return results
