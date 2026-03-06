"""Unit tests for UA-PBR models."""

import unittest
import torch
import sys
sys.path.append('..')

from uapbr.models.autoencoder import PhysicsInformedAutoencoder
from uapbr.models.bayesian_cnn import BayesianCNN
from uapbr.models.standard_cnn import StandardCNN

class TestAutoencoder(unittest.TestCase):
    def setUp(self):
        self.model = PhysicsInformedAutoencoder(latent_dim=128)
        self.batch_size = 4
        self.input_tensor = torch.randn(self.batch_size, 1, 32, 32)
        
    def test_forward_shape(self):
        u_recon, a_recon = self.model(self.input_tensor)
        self.assertEqual(u_recon.shape, self.input_tensor.shape)
        self.assertEqual(a_recon.shape, self.input_tensor.shape)
        
    def test_pde_residual(self):
        u_recon, a_recon = self.model(self.input_tensor)
        residual = self.model.pde_residual(u_recon, a_recon)
        self.assertEqual(residual.shape, (self.batch_size,))
        self.assertTrue(torch.all(residual >= 0))

class TestBayesianCNN(unittest.TestCase):
    def setUp(self):
        self.model = BayesianCNN(num_classes=2, dropout_rate=0.3)
        self.batch_size = 4
        self.input_tensor = torch.randn(self.batch_size, 1, 32, 32)
        
    def test_forward_shape(self):
        logits = self.model(self.input_tensor)
        self.assertEqual(logits.shape, (self.batch_size, 2))
        
    def test_predict_with_uncertainty(self):
        out = self.model.predict_with_uncertainty(self.input_tensor, n_samples=10)
        self.assertIn('probs', out)
        self.assertIn('entropy', out)
        self.assertEqual(out['probs'].shape, (self.batch_size, 2))
        self.assertEqual(out['entropy'].shape, (self.batch_size,))

class TestStandardCNN(unittest.TestCase):
    def setUp(self):
        self.model = StandardCNN(num_classes=2)
        self.batch_size = 4
        self.input_tensor = torch.randn(self.batch_size, 1, 32, 32)
        
    def test_forward_shape(self):
        logits = self.model(self.input_tensor)
        self.assertEqual(logits.shape, (self.batch_size, 2))

if __name__ == '__main__':
    unittest.main()
