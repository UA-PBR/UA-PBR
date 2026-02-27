"""Visualization utilities for paper-quality figures"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_all_results(ae_history, test_scores, best_tau_phy, best_tau_unc, 
                     risks, val_scores, optimizer, save_path='results/'):
    """Generate all paper figures"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    test_phy_scores, test_unc_scores, test_labels, test_preds = test_scores
    val_phy_scores, val_unc_scores, val_labels, val_preds = val_scores
    
    # 1. Training curves
    axes[0,0].plot(ae_history, label='AE Loss', linewidth=2)
    axes[0,0].set_xlabel('Epoch', fontsize=12)
    axes[0,0].set_ylabel('Loss', fontsize=12)
    axes[0,0].set_title('Theorem 2.1: Autoencoder Convergence', fontsize=14)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. PDE residual distribution
    axes[0,1].hist(test_phy_scores, bins=30, alpha=0.7, edgecolor='black')
    axes[0,1].axvline(best_tau_phy, color='r', linestyle='--', linewidth=2, 
                      label=f'τ_phy={best_tau_phy:.2f}')
    axes[0,1].set_xlabel('Physics Score (PDE Residual)', fontsize=12)
    axes[0,1].set_ylabel('Frequency', fontsize=12)
    axes[0,1].set_title('Theorem 2.3: Error Bound via PDE Residual', fontsize=14)
    axes[0,1].legend()
    
    # 3. Uncertainty distribution
    axes[0,2].hist(test_unc_scores, bins=30, alpha=0.7, edgecolor='black')
    axes[0,2].axvline(best_tau_unc, color='r', linestyle='--', linewidth=2,
                      label=f'τ_unc={best_tau_unc:.2f}')
    axes[0,2].set_xlabel('Uncertainty (Entropy)', fontsize=12)
    axes[0,2].set_ylabel('Frequency', fontsize=12)
    axes[0,2].set_title('Theorem 3.1: Bayesian Uncertainty', fontsize=14)
    axes[0,2].legend()
    
    # 4. Risk landscape
    _, _, _, best_p_phy, best_p_unc, risk_matrix = optimizer.optimize(
        val_phy_scores, val_unc_scores, val_labels, val_preds
    )
    
    im = axes[1,0].imshow(risk_matrix.T, origin='lower', 
                          extent=[10, 90, 10, 90], 
                          aspect='auto', cmap='viridis')
    axes[1,0].plot(best_p_phy, best_p_unc, 'r*', markersize=15, label='Optimal')
    axes[1,0].set_xlabel('Physics Threshold Percentile', fontsize=12)
    axes[1,0].set_ylabel('Uncertainty Threshold Percentile', fontsize=12)
    axes[1,0].set_title('Theorem 5.1: Optimal Threshold Selection', fontsize=14)
    plt.colorbar(im, ax=axes[1,0])
    
    # 5. Decision scatter
    correct = test_preds == test_labels
    axes[1,1].scatter(test_unc_scores[correct], test_phy_scores[correct], 
                      c='green', alpha=0.5, s=10, label='Correct')
    axes[1,1].scatter(test_unc_scores[~correct], test_phy_scores[~correct], 
                      c='red', alpha=0.5, s=10, label='Wrong')
    axes[1,1].axhline(best_tau_phy, color='r', linestyle='--', linewidth=2)
    axes[1,1].axvline(best_tau_unc, color='r', linestyle='--', linewidth=2)
    axes[1,1].set_xlabel('Uncertainty', fontsize=12)
    axes[1,1].set_ylabel('Physics Score', fontsize=12)
    axes[1,1].set_title('Theorem 4.3: Rejection Regions', fontsize=14)
    axes[1,1].legend()
    
    # 6. Bootstrap confidence intervals
    axes[1,2].hist(risks, bins=30, alpha=0.7, edgecolor='black')
    axes[1,2].axvline(risks.mean(), color='blue', linestyle='--', linewidth=2,
                      label=f'Mean: {risks.mean():.4f}')
    axes[1,2].set_xlabel('Risk', fontsize=12)
    axes[1,2].set_ylabel('Frequency', fontsize=12)
    axes[1,2].set_title('Theorem 6.1: Bootstrap Confidence Intervals', fontsize=14)
    axes[1,2].legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/uapbr_all_figures.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{save_path}/uapbr_all_figures.png', bbox_inches='tight', dpi=150)
    plt.show()
    
    return fig
