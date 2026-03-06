"""Evaluation metrics for UA-PBR."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class EvaluationMetrics:
    """Comprehensive evaluation metrics for rejection quality."""
    
    @staticmethod
    def rejection_quality(phy_scores, unc_scores, labels, preds, tau_phy, tau_unc, lambda_cost):
        """Evaluate rejection quality.
        
        Args:
            phy_scores: Physics scores (PDE residuals)
            unc_scores: Uncertainty scores (entropy)
            labels: True labels
            preds: Model predictions
            tau_phy: Physics threshold
            tau_unc: Uncertainty threshold
            lambda_cost: Rejection cost
            
        Returns:
            dict: Dictionary of metrics
        """
        reject_phy = phy_scores > tau_phy
        reject_unc = (~reject_phy) & (unc_scores > tau_unc)
        accept = (~reject_phy) & (~reject_unc)
        
        metrics = {
            'acceptance_rate': accept.mean(),
            'rejection_phy': reject_phy.mean(),
            'rejection_unc': reject_unc.mean(),
            'accuracy_accepted': accuracy_score(labels[accept], preds[accept]) if accept.sum() > 0 else 0,
            'f1_accepted': f1_score(labels[accept], preds[accept], average='weighted') if accept.sum() > 0 else 0,
            'accuracy_rejected_phy': accuracy_score(labels[reject_phy], preds[reject_phy]) if reject_phy.sum() > 0 else 0,
            'accuracy_rejected_unc': accuracy_score(labels[reject_unc], preds[reject_unc]) if reject_unc.sum() > 0 else 0,
        }
        
        # Compute risk
        risk = (reject_phy.sum() + reject_unc.sum()) * lambda_cost
        risk += ((~reject_phy) & (~reject_unc) & (preds != labels)).sum()
        metrics['empirical_risk'] = risk / len(labels)
        
        return metrics
    
    @staticmethod
    def calibration_metrics(probs, labels, n_bins=10):
        """Compute calibration metrics (ECE, MCE).
        
        Args:
            probs: Predicted probabilities [n_samples, n_classes]
            labels: True labels
            n_bins: Number of bins for calibration
            
        Returns:
            dict: Calibration metrics
        """
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = (predictions == labels).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            if in_bin.sum() > 0:
                bin_acc = accuracies[in_bin].mean()
                bin_conf = confidences[in_bin].mean()
                bin_diff = np.abs(bin_acc - bin_conf)
                ece += (in_bin.sum() / len(confidences)) * bin_diff
                mce = max(mce, bin_diff)
        
        return {'ece': ece, 'mce': mce}
