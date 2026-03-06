"""Lipschitz constant estimation for Theorem 5.7."""

import torch
import numpy as np

def estimate_lipschitz(model, dataloader, device, n_samples=100):
    """Estimate Lipschitz constant L via finite differences.
    
    Used for Theorem 5.7 risk bound: R ≤ λ + ε₀ + Lδ
    
    Args:
        model: Neural network model
        dataloader: Data loader for sampling inputs
        device: Device to run on
        n_samples: Number of samples for estimation
        
    Returns:
        L: Estimated Lipschitz constant
    """
    model.eval()
    L_estimates = []
    data_iter = iter(dataloader)
    
    for _ in range(n_samples):
        try:
            x, _, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, _, _ = next(data_iter)
        
        x = x.to(device)
        x1 = x + 0.01 * torch.randn_like(x)
        x2 = x + 0.01 * torch.randn_like(x)
        
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
        
        dist_out = torch.norm(out1 - out2, dim=1).mean().item()
        dist_in = torch.norm((x1 - x2).view(x.size(0), -1), dim=1).mean().item()
        
        if dist_in > 1e-8:
            L_estimates.append(dist_out / dist_in)
    
    return np.mean(L_estimates) if L_estimates else 1.0
