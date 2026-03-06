"""Darcy flow dataset generation."""

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

class RockDataset(Dataset):
    """Darcy flow dataset with permeability and pressure fields."""
    
    def __init__(self, n_samples=10000, resolution=32, seed=42):
        self.n_samples = n_samples
        self.resolution = resolution
        np.random.seed(seed)
        self.u, self.a, self.labels = self._generate_data()
        
    def _generate_data(self):
        n, res = self.n_samples, self.resolution
        x = np.linspace(0, 1, res)
        y = np.linspace(0, 1, res)
        X, Y = np.meshgrid(x, y)
        
        u_list, a_list = [], []
        perm_means = []
        
        for i in range(n):
            # Generate permeability
            if np.random.rand() > 0.5:
                perm = np.exp(np.random.randn(res, res) * 0.8 + 1.0)
            else:
                perm = np.exp(np.random.randn(res, res) * 0.4 - 0.5)
            
            perm = (perm - perm.min()) / (perm.max() - perm.min()) * 1.5 + 0.1
            perm_means.append(perm.mean())
            
            # Generate pressure
            dist = ((X - 0.5)**2 + (Y - 0.5)**2)**0.5
            press = 1.0 - 1.2 * dist / (0.5 + 0.5 * perm.mean())
            press += 0.03 * np.random.randn(res, res)
            
            u_list.append(press)
            a_list.append(perm)
        
        u = torch.FloatTensor(np.array(u_list)).unsqueeze(1)
        a = torch.FloatTensor(np.array(a_list)).unsqueeze(1)
        
        # Normalize
        u = (u - u.mean()) / (u.std() + 1e-8)
        a = (a - a.mean()) / (a.std() + 1e-8)
        
        # Binary labels
        perm_means = np.array(perm_means)
        labels = (perm_means > np.median(perm_means)).astype(int)
        
        return u, a, torch.LongTensor(labels)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.u[idx], self.a[idx], self.labels[idx]
