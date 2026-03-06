#!/usr/bin/env python
"""Ablation study: Compare full UA-PBR vs physics-only vs uncertainty-only."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime
import json

from uapbr.config import Config
from uapbr.data.dataset import RockDataset
from uapbr.models.autoencoder import PhysicsInformedAutoencoder
from uapbr.models.bayesian_cnn import BayesianCNN
from uapbr.models.standard_cnn import StandardCNN
from uapbr.training.trainer import train_physics_ae, train_bayesian_cnn
from uapbr.utils.metrics import EvaluationMetrics
from uapbr.utils.thresholds import ThresholdOptimizer

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def main():
    config = Config(
        n_samples=5000,
        n_seeds=5,
        results_dir='./results_ablation',
        figures_dir='./figures_ablation'
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(config.results_dir, exist_ok=True)
    
    results = {
        'full': {'risk': [], 'accept_rate': [], 'acc_accepted': []},
        'physics_only': {'risk': [], 'accept_rate': [], 'acc_accepted': []},
        'uncertainty_only': {'risk': [], 'accept_rate': [], 'acc_accepted': []}
    }
    
    for seed in range(config.n_seeds):
        print(f"\nSeed {seed+1}/{config.n_seeds}")
        set_seed(42 + seed)
        
        # Generate data
        dataset = RockDataset(n_samples=config.n_samples, resolution=config.resolution)
        u, a, labels = dataset.u, dataset.a, dataset.labels
        
        # Split
        n = len(u)
        indices = torch.randperm(n)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]
        
        train_loader = DataLoader(
            TensorDataset(u[train_idx], a[train_idx], labels[train_idx]),
            batch_size=config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(u[val_idx], a[val_idx], labels[val_idx]),
            batch_size=config.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(u[test_idx], a[test_idx], labels[test_idx]),
            batch_size=config.batch_size, shuffle=False
        )
        
        # Train models
        autoencoder = PhysicsInformedAutoencoder(latent_dim=config.latent_dim).to(device)
        autoencoder, _ = train_physics_ae(autoencoder, train_loader, val_loader, config, device)
        
        bayesian_cnn = BayesianCNN(num_classes=2, dropout_rate=config.dropout_rate).to(device)
        bayesian_cnn, _, _ = train_bayesian_cnn(bayesian_cnn, train_loader, val_loader, config, device)
        
        # ... (evaluation code for each variant)
        
        print(f"Seed {seed+1} complete")
    
    print("\nAblation study complete!")
    print(f"Full UA-PBR: Risk = {np.mean(results['full']['risk']):.4f} ± {np.std(results['full']['risk']):.4f}")
    print(f"Physics Only: Risk = {np.mean(results['physics_only']['risk']):.4f} ± {np.std(results['physics_only']['risk']):.4f}")
    print(f"Uncertainty Only: Risk = {np.mean(results['uncertainty_only']['risk']):.4f} ± {np.std(results['uncertainty_only']['risk']):.4f}")

if __name__ == "__main__":
    main()
