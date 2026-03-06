"""Standard CNN baseline without uncertainty."""

import torch
import torch.nn as nn

class StandardCNN(nn.Module):
    """Standard CNN classifier (baseline).
    
    Same architecture as BayesianCNN but without dropout-based
    uncertainty quantification.
    """
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(0.5),  # Regular dropout for training only
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor [batch, 1, H, W]
            
        Returns:
            logits: Classification logits [batch, num_classes]
        """
        return self.net(x)
