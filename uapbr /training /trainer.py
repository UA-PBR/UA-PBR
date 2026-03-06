"""Training functions for UA-PBR components."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy

def train_physics_ae(model, train_loader, val_loader, config, device='cuda'):
    """Train physics-informed autoencoder.
    
    Args:
        model: PhysicsInformedAutoencoder instance
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration object
        device: Device to train on
        
    Returns:
        model: Trained model
        history: Training history
    """
    optimizer = optim.AdamW(model.parameters(), lr=config.initial_lr, 
                           weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    mse = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': []}
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(config.ae_epochs):
        model.train()
        total_loss = 0
        
        loop = tqdm(train_loader, desc=f"AE Epoch {epoch+1}/{config.ae_epochs}")
        for u_batch, a_batch, _ in loop:
            u_batch, a_batch = u_batch.to(device), a_batch.to(device)
            
            optimizer.zero_grad()
            u_recon, a_recon = model(u_batch)
            
            loss_u = mse(u_recon, u_batch)
            loss_a = mse(a_recon, a_batch)
            loss_phy = model.pde_residual(u_recon, a_recon).mean()
            loss = loss_u + loss_a + config.lambda_phy * loss_phy
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for u_batch, a_batch, _ in val_loader:
                u_batch, a_batch = u_batch.to(device), a_batch.to(device)
                u_recon, a_recon = model(u_batch)
                loss_u = mse(u_recon, u_batch)
                loss_a = mse(a_recon, a_batch)
                loss_phy = model.pde_residual(u_recon, a_recon).mean()
                val_loss += (loss_u + loss_a + config.lambda_phy * loss_phy).item()
        
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
        
        if (epoch+1) % 20 == 0:
            print(f"AE Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    model.load_state_dict(best_state)
    return model, history


def train_bayesian_cnn(model, train_loader, val_loader, config, device='cuda'):
    """Train Bayesian CNN.
    
    Args:
        model: BayesianCNN instance
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration object
        device: Device to train on
        
    Returns:
        model: Trained model
        history: Training history
        best_acc: Best validation accuracy
    """
    optimizer = optim.AdamW(model.parameters(), lr=config.initial_lr, 
                           weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
    
    best_acc = 0.0
    best_state = None
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(config.cnn_epochs):
        model.train()
        total_loss = 0
        
        loop = tqdm(train_loader, desc=f"BCNN Epoch {epoch+1}/{config.cnn_epochs}")
        for u_batch, _, labels in loop:
            u_batch, labels = u_batch.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(u_batch)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for u_batch, _, labels in val_loader:
                u_batch, labels = u_batch.to(device), labels.to(device)
                outputs = model(u_batch)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += len(labels)
        
        acc = correct / total
        history['train_loss'].append(total_loss / len(train_loader))
        history['val_acc'].append(acc)
        scheduler.step(acc)
        
        if acc > best_acc:
            best_acc = acc
            best_state = deepcopy(model.state_dict())
        
        if (epoch+1) % 20 == 0:
            print(f"BCNN Epoch {epoch+1}: Val Acc = {acc:.4f}, Best = {best_acc:.4f}")
    
    model.load_state_dict(best_state)
    return model, history, best_acc


def train_standard_cnn(model, train_loader, val_loader, config, device='cuda'):
    """Train standard CNN baseline.
    
    Args:
        model: StandardCNN instance
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration object
        device: Device to train on
        
    Returns:
        model: Trained model
        best_acc: Best validation accuracy
    """
    optimizer = optim.AdamW(model.parameters(), lr=config.initial_lr, 
                           weight_decay=config.weight_decay)
    best_acc = 0.0
    best_state = None
    
    for epoch in range(config.cnn_epochs // 2):
        model.train()
        for u_batch, _, labels in train_loader:
            u_batch, labels = u_batch.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(u_batch), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for u_batch, _, labels in val_loader:
                u_batch, labels = u_batch.to(device), labels.to(device)
                outputs = model(u_batch)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += len(labels)
        
        acc = correct / total
        
        if acc > best_acc:
            best_acc = acc
            best_state = deepcopy(model.state_dict())
        
        if (epoch+1) % 20 == 0:
            print(f"Std CNN Epoch {epoch+1}: Val Acc = {acc:.4f}, Best = {best_acc:.4f}")
    
    model.load_state_dict(best_state)
    return model, best_acc
