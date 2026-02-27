"""Load real experimental rock data from various sources"""

import torch
import numpy as np
import h5py
import os
import requests
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class RealRockDataset:
    """Load real rock permeability data from public repositories"""
    
    def __init__(self, data_dir='./real_rock_data', resolution=32, n_samples=None):
        self.data_dir = data_dir
        self.resolution = resolution
        self.n_samples = n_samples
        os.makedirs(data_dir, exist_ok=True)
        
        self.u, self.a, self.labels = self._load_real_data()
        
    def _load_real_data(self):
        """Try multiple real data sources"""
        
        # Try Digital Rocks Portal first
        try:
            return self._load_digital_rocks()
        except:
            print("Digital Rocks download failed, trying synthetic fallback...")
            return self._load_synthetic_fallback()
    
    def _load_digital_rocks(self):
        """Load from Digital Rocks Portal"""
        # Berea sandstone URL
        url = "https://www.digitalrocksportal.org/media/data/405/5d7f6c215d8fbd647bb8842b/Berea_3D_256x256x256.raw"
        filename = os.path.join(self.data_dir, "berea.raw")
        
        if not os.path.exists(filename):
            print("Downloading Berea sandstone from Digital Rocks...")
            response = requests.get(url, stream=True)
            with open(filename, 'wb') as f:
                for chunk in tqdm(response.iter_content(chunk_size=8192)):
                    f.write(chunk)
        
        # Load 3D volume
        with open(filename, 'rb') as f:
            volume = np.fromfile(f, dtype=np.uint8).reshape(256, 256, 256)
        
        # Extract 2D slices and create pressure fields
        a_list = []
        u_list = []
        
        for z in range(0, 256, 10):
            if self.n_samples and len(a_list) >= self.n_samples:
                break
                
            # Get slice
            perm_slice = volume[:, :, z].astype(np.float32)
            
            # Resize
            from skimage.transform import resize
            perm_resized = resize(perm_slice, (self.resolution, self.resolution), 
                                  mode='reflect', preserve_range=True)
            
            # Normalize to realistic permeability range
            perm_resized = (perm_resized - perm_resized.min()) / (perm_resized.max() - perm_resized.min())
            perm_resized = perm_resized * 0.8 + 0.1
            
            # Create pressure field (approximate solution)
            press = self._approximate_pressure(perm_resized)
            
            a_list.append(perm_resized)
            u_list.append(press)
        
        # Convert to tensors
        a = torch.FloatTensor(np.array(a_list)).unsqueeze(1)
        u = torch.FloatTensor(np.array(u_list)).unsqueeze(1)
        
        # Normalize
        u = (u - u.mean()) / (u.std() + 1e-8)
        a = (a - a.mean()) / (a.std() + 1e-8)
        
        # Create labels
        perm_means = a.mean(dim=[1,2,3])
        labels = (perm_means > perm_means.median()).long()
        
        return u, a, labels
    
    def _approximate_pressure(self, perm):
        """Fast approximate pressure solution"""
        n = len(perm)
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        X, Y = np.meshgrid(x, y)
        
        # Simple parabolic pressure with permeability modulation
        press = 1.0 - ((X-0.5)**2 + (Y-0.5)**2)**0.5 * 1.5
        press = press * (0.5 + 0.5 * perm / perm.mean())
        
        return press
    
    def _load_synthetic_fallback(self):
        """Fallback to synthetic data"""
        from .dataset import RockDataset
        syn_dataset = RockDataset(n_samples=self.n_samples or 500, 
                                  resolution=self.resolution)
        return syn_dataset.u, syn_dataset.a, syn_dataset.labels
    
    def get_data_loaders(self, batch_size=32):
        """Create data loaders"""
        n = len(self.u)
        indices = torch.randperm(n)
        n_train = int(0.6 * n)
        n_val = int(0.2 * n)
        
        train_loader = DataLoader(
            TensorDataset(self.u[indices[:n_train]], 
                         self.a[indices[:n_train]], 
                         self.labels[indices[:n_train]]),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(self.u[indices[n_train:n_train+n_val]], 
                         self.a[indices[n_train:n_train+n_val]], 
                         self.labels[indices[n_train:n_train+n_val]]),
            batch_size=batch_size, shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(self.u[indices[n_train+n_val:]], 
                         self.a[indices[n_train+n_val:]], 
                         self.labels[indices[n_train+n_val:]]),
            batch_size=batch_size, shuffle=False
        )
        return train_loader, val_loader, test_loader
