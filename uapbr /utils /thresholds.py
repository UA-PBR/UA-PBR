"""Threshold optimization utilities."""

import numpy as np

class ThresholdOptimizer:
    """Optimize rejection thresholds via grid search.
    
    Finds optimal thresholds for physics score and uncertainty
    by minimizing empirical risk on validation data.
    """
    
    def __init__(self, lambda_cost=0.3):
        self.lambda_cost = lambda_cost
    
    def compute_risk(self, phy_scores, unc_scores, labels, preds, tau_phy, tau_unc):
        """Compute empirical risk for given thresholds.
        
        Args:
            phy_scores: Physics scores
            unc_scores: Uncertainty scores
            labels: True labels
            preds: Model predictions
            tau_phy: Physics threshold
            tau_unc: Uncertainty threshold
            
        Returns:
            risk: Empirical risk
        """
        reject_phy = phy_scores > tau_phy
        reject_unc = (~reject_phy) & (unc_scores > tau_unc)
        accept = (~reject_phy) & (~reject_unc)
        
        risk = (reject_phy.sum() + reject_unc.sum()) * self.lambda_cost
        risk += ((~reject_phy) & (~reject_unc) & (preds != labels)).sum()
        return risk / len(labels)
    
    def optimize(self, phy_scores, unc_scores, labels, preds, grid_size=40):
        """Grid search for optimal thresholds.
        
        Args:
            phy_scores: Physics scores
            unc_scores: Uncertainty scores
            labels: True labels
            preds: Model predictions
            grid_size: Number of percentile points to try
            
        Returns:
            tau_phy: Optimal physics threshold
            tau_unc: Optimal uncertainty threshold
            best_risk: Minimum risk achieved
            p_phy: Percentile of optimal physics threshold
            p_unc: Percentile of optimal uncertainty threshold
        """
        best_risk = float('inf')
        best_tau_phy = None
        best_tau_unc = None
        best_p_phy, best_p_unc = None, None
        
        for p_phy in np.linspace(10, 90, grid_size):
            for p_unc in np.linspace(10, 90, grid_size):
                tau_phy = np.percentile(phy_scores, p_phy)
                tau_unc = np.percentile(unc_scores, p_unc)
                risk = self.compute_risk(phy_scores, unc_scores, labels, preds, tau_phy, tau_unc)
                
                if risk < best_risk:
                    best_risk = risk
                    best_tau_phy = tau_phy
                    best_tau_unc = tau_unc
                    best_p_phy = p_phy
                    best_p_unc = p_unc
        
        return best_tau_phy, best_tau_unc, best_risk, best_p_phy, best_p_unc
