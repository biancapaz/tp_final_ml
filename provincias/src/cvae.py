"""
cvae.py

Conditional Variational Autoencoder for electricity demand generation.
Works for both architectures:
  Architecture A: x_dim=528, c_dim=267 (system-level, no socioeconomic)
  Architecture B: x_dim=24,  c_dim=28  (province-day, with socioeconomic)

The latent space captures residual variability not explained by conditions.
At generation time: sample z ~ N(0,1), decode with chosen c.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
        self,
        x_dim: int,
        c_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
    ) -> None:
        super().__init__()
        input_dim = x_dim + c_dim
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU()]
            in_dim = h
        self.net    = nn.Sequential(*layers)
        self.mu     = nn.Linear(in_dim, latent_dim)
        self.logvar = nn.Linear(in_dim, latent_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(torch.cat([x, c], dim=-1))
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        c_dim: int,
        hidden_dims: list[int],
        x_dim: int,
    ) -> None:
        super().__init__()
        input_dim = latent_dim + c_dim
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU()]
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, x_dim)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.net(torch.cat([z, c], dim=-1))
        return self.out(h)


class CVAE(nn.Module):
    def __init__(
        self,
        x_dim: int,
        c_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
    ) -> None:
        """
        Conditional Variational Autoencoder.

        x_dim:       dimension of the demand profile vector.
        c_dim:       dimension of the condition vector.
        hidden_dims: list of hidden layer sizes for encoder and decoder.
        latent_dim:  dimension of the latent space z.
        """
        super().__init__()
        self.x_dim      = x_dim
        self.c_dim      = c_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(x_dim, c_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, c_dim, list(reversed(hidden_dims)), x_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * std."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns: (x_recon, mu, logvar)
        """
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, c)
        return x_recon, mu, logvar

    @torch.no_grad()
    def generate(
        self,
        c: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """
        Generate demand profiles by sampling z ~ N(0,1) and decoding.

        c:         condition vector, shape (batch, c_dim) or (c_dim,).
        n_samples: number of samples to generate per condition.
        Returns:   tensor of shape (batch * n_samples, x_dim).
        """
        self.eval()
        if c.dim() == 1:
            c = c.unsqueeze(0)
        c_expanded = c.repeat_interleave(n_samples, dim=0)
        z = torch.randn(len(c_expanded), self.latent_dim, device=c.device)
        return self.decoder(z, c_expanded)


def cvae_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    free_bits: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CVAE loss = reconstruction loss + beta * KL divergence.

    free_bits: minimum KL per latent dimension (prevents posterior collapse).
    Returns: (total_loss, recon_loss, kl_loss)
    """
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")

    # KL divergence per dimension
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

    kl_loss = kl_per_dim.mean()
    total   = recon_loss + beta * kl_loss

    return total, recon_loss, kl_loss
