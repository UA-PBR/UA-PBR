"""Bayesian CNN with Monte Carlo Dropout (Theorem 3.1)"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianCNN(nn.Module):
    """Bayesian Neural Network with Monte Carlo Dropout"""
    
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    
    def predict_with_uncertainty(self, x, n_samples=30):
        """Theorem 3.1: MC Dropout approximates Bayesian inference"""
        self.train()
        probs_list = []
        
        for _ in range(n_samples):
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            probs_list.append(probs.unsqueeze(0))
        
        probs_stack = torch.cat(probs_list, dim=0)
        mean_probs = probs_stack.mean(dim=0)
        
        # Entropy as uncertainty measure
        entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)
        
        return mean_probs, entropy
