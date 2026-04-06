import torch
from torch import nn
import numpy as np

# Rete UNet per prevedere il rumore
class SimpleUNet(nn.Module):
    def __init__(self, time_emb_dim=32):
        super(SimpleUNet, self).__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, time_emb_dim), nn.ReLU())
        self.down = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.mid = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 1, 3, padding=1))

    def forward(self, x, t):
        # NORMALIZZAZIONE DEL TEMPO (assumendo 200 time_steps massimi)
        t_norm = t.float() / 200.0 
        
        t_emb = self.time_embed(t_norm.unsqueeze(1)).view(x.size(0), -1, 1, 1)
        x_encoded = self.down(x)
        x_mid = self.mid(x_encoded + t_emb) # Ora i valori sono bilanciati!
        return self.up(x_mid)

# Classe DDPM completa: coordina la UNet, lo scheduler e il sampling
class DDPM(nn.Module):
    def __init__(self, time_steps=200):
        super(DDPM, self).__init__()
        self.time_steps = time_steps
        self.network = SimpleUNet()
        
        # Scheduler del rumore lineare
        beta = np.linspace(1e-4, 0.02, time_steps)
        alpha = 1 - beta
        self.register_buffer('alpha_hat', torch.from_numpy(np.cumprod(alpha, axis=0)).float())

    def add_noise(self, x, t, epsilon=None):
        """Forward process: q(x_t | x_0)"""
        if epsilon is None: epsilon = torch.randn_like(x)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t]).view(-1, 1, 1, 1)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon

    def forward(self, x, t):
        """La rete DDPM deve prevedere il rumore epsilon"""
        return self.network(x, t)

    @torch.no_grad()
    def sample(self, x_noisy, start_t=None):
        """Reverse process: campionamento iterativo per denoising"""
        if start_t is None: start_t = self.time_steps - 1
        x = x_noisy
        for t in reversed(range(0, start_t + 1)):
            # Previsione del rumore
            t_tensor = torch.full((x.size(0),), t, device=x.device, dtype=torch.long)
            epsilon_theta = self.forward(x, t_tensor)
            
            # Parametri dello scheduler per il tempo t
            alpha_hat = self.alpha_hat[t].view(1, 1, 1, 1)
            alpha = (alpha_hat / (self.alpha_hat[t-1] if t > 0 else 1)).view(1, 1, 1, 1)
            sigma_t = torch.sqrt(1 - alpha).view(1, 1, 1, 1) # Varianza fissa

            # Algoritmo di campionamento (DDPM standard)
            z = torch.randn_like(x) if t > 0 else 0
            x = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * epsilon_theta) + sigma_t * z
        return torch.clamp(x, 0, 1)