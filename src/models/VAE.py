import torch
from torch import nn

class DenoisingVAE(nn.Module):
    def __init__(self, latent_dim=16):
        super(DenoisingVAE, self).__init__()
        # Encoder: da [batch, 1, 28, 28] a [batch, latent_dim]
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent_dim * 2) # mu + log_var
        )
        # Decoder: da [batch, latent_dim] a [batch, 1, 28, 28]
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 28 * 28), nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var) # Reparameterization
        return self.decoder(z), mu, log_var

# La Loss corretta per un Denoising VAE: Reconstruction (Denoising) + KLD
def vae_loss_function(recon, target, mu, log_var, beta=1.0):
    mse = nn.functional.mse_loss(recon, target, reduction='sum')
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return (mse + beta * kld) / recon.size(0) # Loss media per batch