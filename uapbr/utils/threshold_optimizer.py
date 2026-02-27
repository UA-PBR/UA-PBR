"""Optimal threshold selection via risk minimization (Theorem 5.1)"""

import numpy as np

class ThresholdOptimizer:
    """Theorem 5.1: Optimal threshold selection via risk minimization"""
    
    def __init__(self, lambda_cost=0.3):
        self.lambda_cost = lambda_cost
        
    def compute_risk(self, phy_scores, unc_scores, labels, preds, tau_phy, tau_unc):
        """Definition 4.1: Risk with rejection cost"""
        reject_phy = phy_scores > tau_phy
        reject_unc = (~reject_phy) & (unc_scores > tau_unc)
        accept = (~reject_phy) & (~reject_unc)
        
        risk = (reject_phy.sum() + reject_unc.sum()) * self.lambda_cost
        risk += ((~reject_phy) & (~reject_unc) & (preds != labels)).sum()
        
        return risk / len(labels)
    
    def optimize(self, phy_scores, unc_scores, labels, preds, grid_size=20):
        """Grid search for optimal thresholds"""
        best_risk = float('inf')
        best_tau_phy = None
        best_tau_unc = None
        best_p_phy, best_p_unc = None, None
        risk_matrix = np.zeros((grid_size, grid_size))
        
        percentiles = np.linspace(10, 90, grid_size)
        
        for i, p_phy in enumerate(percentiles):
            for j, p_unc in enumerate(percentiles):
                tau_phy = np.percentile(phy_scores, p_phy)
                tau_unc = np.percentile(unc_scores, p_unc)
                
                risk = self.compute_risk(phy_scores, unc_scores, labels, preds, tau_phy, tau_unc)
                risk_matrix[i, j] = risk
                
                if risk < best_risk:
                    best_risk = risk
                    best_tau_phy = tau_phy
                    best_tau_unc = tau_unc
                    best_p_phy = p_phy
                    best_p_unc = p_unc
        
        return best_tau_phy, best_tau_unc, best_risk, best_p_phy, best_p_unc, risk_matrix
