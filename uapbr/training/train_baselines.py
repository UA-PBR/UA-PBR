"""Training functions for baseline models"""

import torch
import torch.nn.functional as F
import torch.optim as optim

def train_standard_cnn(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cuda'):
    """Train standard CNN baseline"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        for u_batch, _, labels in train_loader:
            u_batch, labels = u_batch.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(u_batch), labels)
            loss.backward()
            optimizer.step()
        
        # Print validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for u_batch, _, labels in val_loader:
                    u_batch, labels = u_batch.to(device), labels.to(device)
                    correct += (model(u_batch).argmax(1) == labels).sum().item()
                    total += len(labels)
            print(f"Epoch {epoch+1}: Val Acc = {correct/total:.4f}")
    
    return model
