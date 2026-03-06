"""Corruption generation utilities."""

import torch
import torch.nn.functional as F
import numpy as np

class CorruptionGenerator:
    """Generate various types of corruption for testing robustness."""
    
    @staticmethod
    def gaussian_noise(u, severity):
        """Add Gaussian noise."""
        noise = torch.randn_like(u) * severity
        return u + noise
    
    @staticmethod
    def salt_pepper(u, severity):
        """Add salt and pepper noise."""
        mask = torch.rand_like(u) < severity
        u_corr = u.clone()
        u_corr[mask] = torch.sign(u_corr[mask]) * 2.0
        return u_corr
    
    @staticmethod
    def structured_artifacts(u, severity):
        """Add structured block artifacts."""
        u_corr = u.clone()
        b = 8
        n_blocks = int(severity * 5)
        for _ in range(n_blocks):
            i = np.random.randint(0, u.shape[2] - b)
            j = np.random.randint(0, u.shape[3] - b)
            u_corr[:, :, i:i+b, j:j+b] = torch.randn_like(u_corr[:, :, i:i+b, j:j+b]) * 0.5
        return u_corr
    
    @staticmethod
    def physics_violating(u, severity):
        """Add physics-violating perturbations."""
        noise = torch.randn_like(u) * severity
        kernel = torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], 
                               dtype=torch.float32) / 8.0
        kernel = kernel.to(u.device)
        high_freq = F.conv2d(noise, kernel, padding=1)
        return u + high_freq * severity
    
    @classmethod
    def apply(cls, u, corr_type, severity):
        """Apply corruption by type."""
        if corr_type == 'gaussian':
            return cls.gaussian_noise(u, severity)
        elif corr_type == 'salt_pepper':
            return cls.salt_pepper(u, severity)
        elif corr_type == 'structured':
            return cls.structured_artifacts(u, severity)
        elif corr_type == 'physics_violating':
            return cls.physics_violating(u, severity)
        else:
            return u
