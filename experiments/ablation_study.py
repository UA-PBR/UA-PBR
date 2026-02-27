"""Ablation study to validate each component"""

import torch
import numpy as np
from uapbr.data.dataset import RockDataset
from uapbr.models.physics_ae import PhysicsInformedAutoencoder
from uapbr.models.bayesian_cnn import BayesianCNN
from uapbr.training.train_ae import train_physics_ae
from uapbr.training.train_bcnn import train_bayesian_cnn
from uapbr.utils.metrics import EvaluationMetrics
from uapbr.utils.threshold_optimizer import ThresholdOptimizer
import matplotlib.pyplot as plt

def run_ablation_study(config):
    """Compare full UA-PBR vs components"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    dataset = RockDataset(n_samples=config['n_samples'], resolution=config['resolution'])
    train_loader, val_loader, test_loader = dataset.get_data_loaders(config['batch_size'])
    
    # Train models
    phy_ae = PhysicsInformedAutoencoder().to(device)
    phy_ae, _ = train_physics_ae(phy_ae, train_loader, val_loader, epochs=config['ae_epochs'])
    
    bcnn = BayesianCNN().to(device)
    bcnn, _ = train_bayesian_cnn(bcnn, train_loader, val_loader, epochs=config['cnn_epochs'])
    
    # Collect scores
    phy_ae.eval()
    bcnn.eval()
    
    test_phy_scores, test_unc_scores, test_labels, test_preds = [], [], [], []
    
    with torch.no_grad():
        for u_batch, a_batch, labels in test_loader:
            u_batch = u_batch.to(device)
            
            u_recon, a_recon = phy_ae(u_batch)
            phys_score = phy_ae.pde_residual(u_recon, a_recon)
            test_phy_scores.extend(phys_score.cpu().numpy())
            
            probs, entropy = bcnn.predict_with_uncertainty(u_batch, n_samples=config['n_mc_samples'])
            test_unc_scores.extend(entropy.cpu().numpy())
            test_preds.extend(probs.argmax(1).cpu().numpy())
            test_labels.extend(labels.numpy())
    
    # Normalize
    test_phy_scores = np.array(test_phy_scores)
    test_unc_scores = np.array(test_unc_scores)
    test_phy_scores = (test_phy_scores - test_phy_scores.min()) / (test_phy_scores.max() - test_phy_scores.min() + 1e-8)
    test_unc_scores = (test_unc_scores - test_unc_scores.min()) / (test_unc_scores.max() - test_unc_scores.min() + 1e-8)
    
    # Test different configurations
    configs = [
        {'name': 'Full UA-PBR', 'use_phy': True, 'use_unc': True},
        {'name': 'Physics Only', 'use_phy': True, 'use_unc': False},
        {'name': 'Uncertainty Only', 'use_phy': False, 'use_unc': True},
        {'name': 'No Rejection', 'use_phy': False, 'use_unc': False}
    ]
    
    results = []
    tau_phy, tau_unc = 0.5, 0.5  # Fixed thresholds for comparison
    
    for cfg in configs:
        tau_phy_adj = tau_phy if cfg['use_phy'] else np.inf
        tau_unc_adj = tau_unc if cfg['use_unc'] else np.inf
        
        metrics = EvaluationMetrics.rejection_quality(
            test_phy_scores, test_unc_scores, np.array(test_labels), np.array(test_preds),
            tau_phy_adj, tau_unc_adj, config['lambda_cost']
        )
        results.append(metrics)
    
    # Plot ablation results
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(configs))
    width = 0.25
    
    acc_vals = [r['accuracy_accepted'] for r in results]
    risk_vals = [r['empirical_risk'] for r in results]
    accept_vals = [r['acceptance_rate'] for r in results]
    
    ax.bar(x - width, accept_vals, width, label='Acceptance Rate')
    ax.bar(x, acc_vals, width, label='Accuracy')
    ax.bar(x + width, risk_vals, width, label='Risk')
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Value')
    ax.set_title('Ablation Study')
    ax.set_xticks(x)
    ax.set_xticklabels([cfg['name'] for cfg in configs], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ablation_results.pdf', bbox_inches='tight')
    plt.show()
    
    return results
