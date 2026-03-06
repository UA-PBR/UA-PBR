"""Unit tests for UA-PBR metrics."""

import unittest
import numpy as np
import sys
sys.path.append('..')

from uapbr.utils.metrics import EvaluationMetrics
from uapbr.utils.thresholds import ThresholdOptimizer

class TestEvaluationMetrics(unittest.TestCase):
    def setUp(self):
        self.n_samples = 100
        self.phy_scores = np.random.rand(self.n_samples)
        self.unc_scores = np.random.rand(self.n_samples)
        self.labels = np.random.randint(0, 2, self.n_samples)
        self.preds = np.random.randint(0, 2, self.n_samples)
        self.tau_phy = 0.5
        self.tau_unc = 0.5
        self.lambda_cost = 0.3
        
    def test_rejection_quality(self):
        metrics = EvaluationMetrics.rejection_quality(
            self.phy_scores, self.unc_scores, self.labels, self.preds,
            self.tau_phy, self.tau_unc, self.lambda_cost
        )
        
        self.assertIn('acceptance_rate', metrics)
        self.assertIn('rejection_phy', metrics)
        self.assertIn('rejection_unc', metrics)
        self.assertIn('empirical_risk', metrics)
        
        # Check that rates sum to 1
        self.assertAlmostEqual(
            metrics['acceptance_rate'] + metrics['rejection_phy'] + metrics['rejection_unc'],
            1.0, places=5
        )
        
    def test_calibration_metrics(self):
        probs = np.random.rand(self.n_samples, 2)
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        cal = EvaluationMetrics.calibration_metrics(probs, self.labels)
        self.assertIn('ece', cal)
        self.assertIn('mce', cal)
        self.assertTrue(0 <= cal['ece'] <= 1)
        self.assertTrue(0 <= cal['mce'] <= 1)

class TestThresholdOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = ThresholdOptimizer(lambda_cost=0.3)
        self.n_samples = 100
        self.phy_scores = np.random.rand(self.n_samples)
        self.unc_scores = np.random.rand(self.n_samples)
        self.labels = np.random.randint(0, 2, self.n_samples)
        self.preds = np.random.randint(0, 2, self.n_samples)
        
    def test_compute_risk(self):
        risk = self.optimizer.compute_risk(
            self.phy_scores, self.unc_scores, self.labels, self.preds,
            0.5, 0.5
        )
        self.assertTrue(0 <= risk <= 1)
        
    def test_optimize(self):
        tau_phy, tau_unc, best_risk, p_phy, p_unc = self.optimizer.optimize(
            self.phy_scores, self.unc_scores, self.labels, self.preds,
            grid_size=10
        )
        
        self.assertIsNotNone(tau_phy)
        self.assertIsNotNone(tau_unc)
        self.assertTrue(0 <= best_risk <= 1)

if __name__ == '__main__':
    unittest.main()
