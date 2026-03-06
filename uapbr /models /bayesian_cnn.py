"""Bayesian CNN with Monte Carlo Dropout for uncertainty quantification."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianCNN(nn.Module):
    """Bayesian Convolutional Neural Network with MC Dropout.
    
    Uses dropout at inference time to approximate Bayesian inference,
    providing uncertainty estimates via predictive entropy.
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor [batch, 1, H, W]
            
        Returns:
            logits: Classification logits [batch, num_classes]
        """
        x = self.features(x)
        return self.classifier(x)
    
    def predict_with_uncertainty(self, x, n_samples=50, temperature=1.2):
        """MC Dropout prediction with uncertainty estimates.
        
        Args:
            x: Input tensor [batch, 1, H, W]
            n_samples: Number of Monte Carlo samples
            temperature: Temperature scaling for calibration
            
        Returns:
            dict with keys:
                probs: Mean predictive probabilities
                entropy: Predictive entropy (uncertainty)
                samples: All MC samples
        """
        self.train()  # Enable dropout
        probs_list = []
        
        for _ in range(n_samples):
            logits = self.forward(x) / temperature
            probs = F.softmax(logits, dim=-1)
            probs_list.append(probs.unsqueeze(0))
        
        probs_stack = torch.cat(probs_list, dim=0)
        mean_probs = probs_stack.mean(dim=0)
        
        # Predictive entropy: H[p] = -∑ p_k log p_k
        entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)
        
        return {
            'probs': mean_probs,
            'entropy': entropy,
            'samples': probs_stack
        }
