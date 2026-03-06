"""Physics-informed autoencoder with PDE residual."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedAutoencoder(nn.Module):
    """Autoencoder that learns to reconstruct pressure and permeability fields
    while respecting the governing PDE (Darcy's law).
    """
    
    def __init__(self, latent_dim=256):
        super().__init__()
        
        # Encoder: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.encoder_fc = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder for pressure
        self.decoder_fc = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.dec_conv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU()
        )
        
        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU()
        )
        
        self.dec_conv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU()
        )
        
        self.dec_conv4 = nn.Conv2d(32, 1, 3, padding=1)
        
        # Decoder for permeability
        self.decoder_a_fc = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.dec_a_conv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU()
        )
        
        self.dec_a_conv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU()
        )
        
        self.dec_a_conv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU()
        )
        
        self.dec_a_conv4 = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Softplus()
        )
        
    def pde_residual(self, u, a, dx=1.0):
        """Compute PDE residual: -∇·(a∇u) - f = 0
        
        Args:
            u: Pressure field [batch, 1, H, W]
            a: Permeability field [batch, 1, H, W]
            dx: Grid spacing
            
        Returns:
            PDE residual magnitude per sample
        """
        # Pad for boundaries
        u_pad = F.pad(u, (1, 1, 1, 1), mode='replicate')
        a_pad = F.pad(a, (1, 1, 1, 1), mode='replicate')
        
        # Central differences for gradients
        u_x = (u_pad[:, :, 1:-1, 2:] - u_pad[:, :, 1:-1, :-2]) / (2 * dx)
        u_y = (u_pad[:, :, 2:, 1:-1] - u_pad[:, :, :-2, 1:-1]) / (2 * dx)
        
        a_center = a_pad[:, :, 1:-1, 1:-1]
        flux_x = a_center * u_x
        flux_y = a_center * u_y
        
        # Divergence of flux
        div_flux_x = (flux_x[:, :, 1:-1, 2:] - flux_x[:, :, 1:-1, :-2]) / (2 * dx)
        div_flux_y = (flux_y[:, :, 2:, 1:-1] - flux_y[:, :, :-2, 1:-1]) / (2 * dx)
        div_flux = div_flux_x + div_flux_y
        
        # Source term f = 1 (constant)
        source = torch.ones_like(div_flux)
        
        # Residual: -∇·(a∇u) - f
        residual = -div_flux - source
        return residual.abs().mean(dim=(1, 2, 3))
    
    def forward(self, u):
        """Forward pass through autoencoder.
        
        Args:
            u: Input pressure field [batch, 1, H, W]
            
        Returns:
            u_recon: Reconstructed pressure field
            a_recon: Reconstructed permeability field
        """
        # Encode
        e1 = self.enc_conv1(u)
        e2 = self.enc_conv2(e1)
        e3 = self.enc_conv3(e2)
        e4 = self.enc_conv4(e3)
        
        z = self.encoder_fc(e4.view(e4.size(0), -1))
        
        # Decode pressure
        d = self.decoder_fc(z).view(-1, 256, 4, 4)
        d1 = self.dec_conv1(d)
        d2 = self.dec_conv2(d1)
        d3 = self.dec_conv3(d2)
        u_recon = self.dec_conv4(d3)
        
        # Decode permeability
        d_a = self.decoder_a_fc(z).view(-1, 256, 4, 4)
        d_a1 = self.dec_a_conv1(d_a)
        d_a2 = self.dec_a_conv2(d_a1)
        d_a3 = self.dec_a_conv3(d_a2)
        a_recon = self.dec_a_conv4(d_a3)
        
        return u_recon, a_recon
