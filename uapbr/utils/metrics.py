"""Evaluation metrics for rejection systems (Theorem 4.3, 6.1)"""

import numpy as np
from sklearn.metrics import accuracy_score

class EvaluationMetrics:
    """Theorem 4.3: Risk bounds and rejection quality"""
    
    @staticmethod
    def rejection_quality(phy_scores, unc_scores, labels, preds, tau_phy, tau_unc, lambda_cost):
        """Compute rejection metrics"""
        reject_phy = phy_scores > tau_phy
        reject_unc = (~reject_phy) & (unc_scores > tau_unc)
        accept = (~reject_phy) & (~reject_unc)
        
        metrics = {
            'acceptance_rate': accept.mean(),
            'rejection_phy': reject_phy.mean(),
            'rejection_unc': reject_unc.mean(),
            'accuracy_accepted': accuracy_score(labels[accept], preds[accept]) if accept.sum() > 0 else 0,
            'accuracy_rejected_phy': accuracy_score(labels[reject_phy], preds[reject_phy]) if reject_phy.sum() > 0 else 0,
            'accuracy_rejected_unc': accuracy_score(labels[reject_unc], preds[reject_unc]) if reject_unc.sum() > 0 else 0,
        }
        
        # Theorem 4.3: Empirical risk and theoretical bound
        risk = (reject_phy.sum() + reject_unc.sum()) * lambda_cost
        risk += ((~reject_phy) & (~reject_unc) & (preds != labels)).sum()
        metrics['empirical_risk'] = risk / len(labels)
        metrics['theoretical_bound'] = lambda_cost + reject_phy.mean() + reject_unc.mean()
        
        return metrics
    
    @staticmethod
    def bootstrap_ci(scores, n_bootstrap=1000, ci=95):
        """Theorem 6.1: Confidence intervals via bootstrap"""
        means = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(scores), len(scores), replace=True)
            means.append(scores[idx].mean())
        
        lower = np.percentile(means, (100 - ci) / 2)
        upper = np.percentile(means, 100 - (100 - ci) / 2)
        return lower, upper
