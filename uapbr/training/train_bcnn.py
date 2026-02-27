"""Training functions for Bayesian CNN"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

def train_bayesian_cnn(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cuda'):
    """Train Bayesian CNN"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = []
    
    for epoch in range(epochs):
        model.train()
        for u_batch, _, labels in train_loader:
            u_batch, labels = u_batch.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(u_batch)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for u_batch, _, labels in val_loader:
                u_batch, labels = u_batch.to(device), labels.to(device)
                logits = model(u_batch)
                correct += (logits.argmax(1) == labels).sum().item()
                total += len(labels)
        
        acc = correct / total
        history.append(acc)
        print(f"Epoch {epoch+1}: Val Acc = {acc:.4f}")
    
    return model, history
