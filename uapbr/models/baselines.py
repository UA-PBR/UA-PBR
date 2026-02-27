"""Baseline models for comparison"""

import torch
import torch.nn as nn
from .bayesian_cnn import BayesianCNN

class StandardCNN(nn.Module):
    """Standard CNN without uncertainty"""
    
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        return self.network(x)

class MCDropoutOnly(BayesianCNN):
    """MC Dropout without physics filter (same architecture)"""
    pass
