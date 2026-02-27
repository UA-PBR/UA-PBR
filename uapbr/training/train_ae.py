"""Training functions for Physics-Informed Autoencoder"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_physics_ae(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cuda'):
    """Train physics-informed autoencoder"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    history = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for u_batch, a_batch, _ in tqdm(train_loader, desc=f"AE Epoch {epoch+1}"):
            u_batch, a_batch = u_batch.to(device), a_batch.to(device)
            
            u_recon, a_recon = model(u_batch)
            
            loss_u = mse(u_recon, u_batch)
            loss_a = mse(a_recon, a_batch)
            loss_phy = model.pde_residual(u_recon, a_recon).mean()
            
            loss = loss_u + loss_a + 0.1 * loss_phy
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        history.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    return model, history
