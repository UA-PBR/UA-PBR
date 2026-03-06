"""Custom loss functions for UA-PBR."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedLoss(nn.Module):
    """Combined loss with reconstruction and PDE residual terms."""
    
    def __init__(self, lambda_phy=0.1):
        super().__init__()
        self.lambda_phy = lambda_phy
        self.mse = nn.MSELoss()
        
    def forward(self, u_recon, u_true, a_recon, a_true, pde_residual):
        """Compute total loss.
        
        Args:
            u_recon: Reconstructed pressure
            u_true: True pressure
            a_recon: Reconstructed permeability
            a_true: True permeability
            pde_residual: PDE residual
            
        Returns:
            total_loss: Weighted combination of losses
            loss_dict: Dictionary of individual losses
        """
        loss_u = self.mse(u_recon, u_true)
        loss_a = self.mse(a_recon, a_true)
        loss_phy = pde_residual.mean()
        
        total = loss_u + loss_a + self.lambda_phy * loss_phy
        
        return total, {
            'reconstruction_u': loss_u.item(),
            'reconstruction_a': loss_a.item(),
            'physics': loss_phy.item(),
            'total': total.item()
        }


class UncertaintyAwareLoss(nn.Module):
    """Loss function for Bayesian CNN with uncertainty regularization."""
    
    def __init__(self, num_classes=2, label_smoothing=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        
    def forward(self, logits, labels, variances=None):
        """Compute cross-entropy loss with optional variance regularization.
        
        Args:
            logits: Classification logits
            labels: True labels
            variances: Predictive variances (optional)
            
        Returns:
            loss: Total loss
        """
        # Cross-entropy with optional label smoothing
        if self.label_smoothing > 0:
            loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        else:
            loss = F.cross_entropy(logits, labels)
        
        # Add variance regularization if provided
        if variances is not None:
            loss += 0.01 * variances.mean()
            
        return loss
