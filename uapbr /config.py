"""Configuration management for UA-PBR."""

class Config:
    """Configuration class with default production settings."""
    
    def __init__(
        self,
        n_samples=10000,
        resolution=32,
        batch_size=64,
        ae_epochs=150,
        cnn_epochs=200,
        initial_lr=1e-3,
        weight_decay=1e-4,
        latent_dim=256,
        dropout_rate=0.3,
        n_mc_samples=50,
        lambda_cost=0.3,
        lambda_phy=0.1,
        **kwargs
    ):
        self.n_samples = n_samples
        self.resolution = resolution
        self.batch_size = batch_size
        self.ae_epochs = ae_epochs
        self.cnn_epochs = cnn_epochs
        self.initial_lr = initial_lr
        self.weight_decay = weight_decay
        self.grad_clip = 1.0
        self.patience = 25
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.n_mc_samples = n_mc_samples
        self.temperature = 1.2
        self.lambda_cost = lambda_cost
        self.lambda_phy = lambda_phy
        self.val_split = 0.15
        self.test_split = 0.15
        self.num_classes = 2
        self.corruption_types = ['gaussian', 'salt_pepper', 'structured', 'physics_violating']
        self.severities = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        # Update with any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self):
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
