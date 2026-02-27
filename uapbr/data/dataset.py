"""Realistic rock dataset generation and loading"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split

class RockDataset:
    """Generate realistic rock permeability and pressure data"""
    
    def __init__(self, n_samples=1000, resolution=32, seed=42):
        self.n_samples = n_samples
        self.resolution = resolution
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.u, self.a, self.labels = self._generate_data()
        self._split_data()
    
    def _generate_data(self):
        """Generate synthetic but realistic rock data"""
        n, res = self.n_samples, self.resolution
        
        # Create coordinate grid
        x = np.linspace(0, 1, res)
        y = np.linspace(0, 1, res)
        X, Y = np.meshgrid(x, y)
        
        u_list, a_list = [], []
        
        for i in range(n):
            # Random permeability field with spatial correlation
            perm = np.zeros((res, res))
            for k in range(1, 4):
                phase_x = np.random.rand() * 2 * np.pi
                phase_y = np.random.rand() * 2 * np.pi
                perm += np.random.randn() * np.sin(2*np.pi*k*X + phase_x) * np.cos(2*np.pi*k*Y + phase_y)
            
            # Transform to log-normal (realistic permeability)
            perm = np.exp(perm * 0.5)
            perm = (perm - perm.min()) / (perm.max() - perm.min()) * 0.8 + 0.1
            
            # Solve Darcy's law approximately
            dist = ((X - 0.5)**2 + (Y - 0.5)**2)**0.5
            press = 1.0 - 1.5 * dist + 0.1 * np.random.randn(res, res)
            
            u_list.append(press)
            a_list.append(perm)
        
        # Convert to tensors
        u = torch.FloatTensor(np.array(u_list)).unsqueeze(1)
        a = torch.FloatTensor(np.array(a_list)).unsqueeze(1)
        
        # Normalize
        u = (u - u.mean()) / (u.std() + 1e-8)
        a = (a - a.mean()) / (a.std() + 1e-8)
        
        # Create binary labels based on mean permeability
        perm_means = a.mean(dim=[1,2,3])
        labels = (perm_means > perm_means.median()).long()
        
        return u, a, labels
    
    def _split_data(self):
        """Split into train/val/test"""
        indices = torch.randperm(self.n_samples)
        n_train = int(0.6 * self.n_samples)
        n_val = int(0.2 * self.n_samples)
        
        self.train_idx = indices[:n_train]
        self.val_idx = indices[n_train:n_train+n_val]
        self.test_idx = indices[n_train+n_val:]
    
    def get_data_loaders(self, batch_size=32):
        """Create data loaders for train/val/test"""
        train_loader = DataLoader(
            TensorDataset(self.u[self.train_idx], 
                         self.a[self.train_idx], 
                         self.labels[self.train_idx]),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(self.u[self.val_idx], 
                         self.a[self.val_idx], 
                         self.labels[self.val_idx]),
            batch_size=batch_size, shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(self.u[self.test_idx], 
                         self.a[self.test_idx], 
                         self.labels[self.test_idx]),
            batch_size=batch_size, shuffle=False
        )
        return train_loader, val_loader, test_loader
