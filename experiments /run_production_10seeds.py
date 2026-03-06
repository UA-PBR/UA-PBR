#!/usr/bin/env python
"""Production run with 10 seeds for statistical power."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime
import json
import pickle

from uapbr.config import Config
from uapbr.data.dataset import RockDataset
from uapbr.data.corruption import CorruptionGenerator
from uapbr.models.autoencoder import PhysicsInformedAutoencoder
from uapbr.models.bayesian_cnn import BayesianCNN
from uapbr.models.standard_cnn import StandardCNN
from uapbr.training.trainer import train_physics_ae, train_bayesian_cnn, train_standard_cnn
from uapbr.utils.metrics import EvaluationMetrics
from uapbr.utils.thresholds import ThresholdOptimizer
from uapbr.utils.lipschitz import estimate_lipschitz
from uapbr.visualization.plotter import generate_figure

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Configuration
    config = Config(
        n_samples=10000,
        n_seeds=10,
        results_dir='./results_production_10seeds',
        figures_dir='./figures_production_10seeds'
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.figures_dir, exist_ok=True)
    
    # Save config
    with open(f"{config.results_dir}/config.json", 'w') as f:
        json.dump(config.to_dict(), f, indent=4)
    
    # Initialize metrics storage
    all_metrics = {}
    for ct in config.corruption_types:
        for sev in config.severities:
            key = f"{ct}_{sev}"
            all_metrics[key] = {'uapbr_risk': [], 'std_risk': [], 'accept_rate': []}
    all_metrics['clean'] = {'uapbr_risk': [], 'std_risk': [], 'accept_rate': []}
    
    total_start = datetime.now()
    
    # Run seeds
    for seed in range(config.n_seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed+1}/{config.n_seeds} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print('='*60)
        
        set_seed(42 + seed)
        
        # Generate data
        dataset = RockDataset(n_samples=config.n_samples, resolution=config.resolution)
        u, a, labels = dataset.u, dataset.a, dataset.labels
        
        # Split
        n = len(u)
        indices = torch.randperm(n)
        n_train = int(n * (1 - config.val_split - config.test_split))
        n_val = int(n * config.val_split)
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
        bayesian_cnn, _, best_acc = train_bayesian_cnn(bayesian_cnn, train_loader, val_loader, config, device)
        print(f"Best accuracy: {best_acc:.4f}")
        
        standard_cnn = StandardCNN(num_classes=2).to(device)
        standard_cnn, _ = train_standard_cnn(standard_cnn, train_loader, val_loader, config, device)
        
        # Evaluate
        # ... (evaluation code here)
        
        print(f"Seed {seed+1} complete")
    
    print(f"\nTotal time: {datetime.now() - total_start}")
    print("✅ Production run complete!")

if __name__ == "__main__":
    main()
