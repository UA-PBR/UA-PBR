"""Physics-Informed Autoencoder with PDE residual"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedAutoencoder(nn.Module):
    """Autoencoder with PDE residual loss (Theorem 2.1, 2.3)"""
    
    def __init__(self, latent_dim=64):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
            nn.Linear(128 * 2 * 2, latent_dim),
        )
        
        # Pressure decoder
        self.decoder_u = nn.Sequential(
            nn.Linear(latent_dim, 128 * 2 * 2), nn.ReLU(),
            nn.Unflatten(1, (128, 2, 2)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )
        
        # Permeability decoder
        self.decoder_a = nn.Sequential(
            nn.Linear(latent_dim, 128 * 2 * 2), nn.ReLU(),
            nn.Unflatten(1, (128, 2, 2)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Softplus()
        )
    
    def pde_residual(self, u, a, dx=1.0):
        """Theorem 2.3: PDE residual bounds prediction error"""
        # Pad for boundaries
        u_pad = F.pad(u, (1, 1, 1, 1), mode='replicate')
        a_pad = F.pad(a, (1, 1, 1, 1), mode='replicate')
        
        # Gradients (central difference)
        u_x = (u_pad[:, :, 1:-1, 2:] - u_pad[:, :, 1:-1, :-2]) / (2 * dx)
        u_y = (u_pad[:, :, 2:, 1:-1] - u_pad[:, :, :-2, 1:-1]) / (2 * dx)
        
        # Flux
        a_center = a_pad[:, :, 1:-1, 1:-1]
        flux_x = a_center * u_x
        flux_y = a_center * u_y
        
        # Divergence
        div_flux_x = (flux_x[:, :, 1:-1, 2:] - flux_x[:, :, 1:-1, :-2]) / (2 * dx)
        div_flux_y = (flux_y[:, :, 2:, 1:-1] - flux_y[:, :, :-2, 1:-1]) / (2 * dx)
        div_flux = div_flux_x + div_flux_y
        
        # Source term
        source = torch.ones_like(div_flux)
        
        # Residual: -div(a∇u) - f
        residual = -div_flux - source
        return residual.abs().mean(dim=(1,2,3))
    
    def forward(self, u):
        z = self.encoder(u)
        return self.decoder_u(z), self.decoder_a(z)
