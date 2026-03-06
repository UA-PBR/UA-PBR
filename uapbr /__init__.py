"""
UA-PBR: Uncertainty-Aware Classifier with Physics-Based Rejection
A unified framework for robust scientific machine learning.
"""

__version__ = "1.0.0"
__author__ = "Mohsen Mostafa"
__email__ = "mohsen.mostafa.ai@outlook.com"

from uapbr.config import Config
from uapbr.models.autoencoder import PhysicsInformedAutoencoder
from uapbr.models.bayesian_cnn import BayesianCNN
from uapbr.models.standard_cnn import StandardCNN
from uapbr.training.trainer import Trainer
from uapbr.utils.metrics import EvaluationMetrics
from uapbr.utils.thresholds import ThresholdOptimizer

class UA_PBR:
    """Main UA-PBR framework class."""
    
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config(**kwargs)
        self.config = config
        self.autoencoder = None
        self.bayesian_cnn = None
        self.standard_cnn = None
        self.thresholds = None
        
    def fit(self, train_loader, val_loader):
        """Train all components."""
        from uapbr.training.trainer import train_physics_ae, train_bayesian_cnn, train_standard_cnn
        
        print("🔧 Training Physics Autoencoder...")
        self.autoencoder = PhysicsInformedAutoencoder(latent_dim=self.config.latent_dim)
        self.autoencoder, _ = train_physics_ae(
            self.autoencoder, train_loader, val_loader, self.config
        )
        
        print("🔧 Training Bayesian CNN...")
        self.bayesian_cnn = BayesianCNN(
            num_classes=self.config.num_classes,
            dropout_rate=self.config.dropout_rate
        )
        self.bayesian_cnn, _, _ = train_bayesian_cnn(
            self.bayesian_cnn, train_loader, val_loader, self.config
        )
        
        print("🔧 Training Standard CNN...")
        self.standard_cnn = StandardCNN(num_classes=self.config.num_classes)
        self.standard_cnn, _ = train_standard_cnn(
            self.standard_cnn, train_loader, val_loader, self.config
        )
        
        return self
    
    def evaluate(self, test_loader):
        """Evaluate on test set."""
        # Implementation here
        pass
