import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encodes a demand profile and its conditioning vector into a
    distribution over the latent space (mu and log_var).
    """

    def __init__(self, profile_dim: int, condition_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(profile_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.log_var_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the encoder.

        Args:
            x: Demand profile tensor of shape (batch, profile_dim).
            c: Conditioning vector tensor of shape (batch, condition_dim).

        Returns:
            Tuple of (mu, log_var), each of shape (batch, latent_dim).
        """
        xc = torch.cat([x, c], dim=1)
        h = self.net(xc)
        return self.mu_layer(h), self.log_var_layer(h)


class Decoder(nn.Module):
    """
    Decodes a latent vector and its conditioning vector into a
    reconstructed demand profile.
    """

    def __init__(self, latent_dim: int, condition_dim: int, hidden_dim: int, profile_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, profile_dim),
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            z: Latent vector of shape (batch, latent_dim).
            c: Conditioning vector of shape (batch, condition_dim).

        Returns:
            Reconstructed profile of shape (batch, profile_dim).
        """
        zc = torch.cat([z, c], dim=1)
        return self.net(zc)


class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder for daily electricity demand profiles.

    The model takes a 24-hour demand profile and a conditioning vector C
    (calendar and weather variables) and learns to generate new profiles
    consistent with those conditions.
    """

    def __init__(
        self,
        profile_dim: int = 24,
        condition_dim: int = 10,
        hidden_dim: int = 128,
        latent_dim: int = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(profile_dim, condition_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, condition_dim, hidden_dim, profile_dim)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: sample z = mu + eps * std where eps ~ N(0, I).

        This allows gradients to flow through the sampling operation during training.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode, sample, decode.

        Returns:
            Tuple of (reconstructed profile, mu, log_var).
        """
        mu, log_var = self.encoder(x, c)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decoder(z, c)
        return x_reconstructed, mu, log_var

    def generate(self, c: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Generate demand profiles by sampling from the prior N(0, I).

        Args:
            c: Conditioning vector of shape (n_samples, condition_dim).
            n_samples: Number of profiles to generate.

        Returns:
            Generated profiles of shape (n_samples, profile_dim).
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=next(self.parameters()).device)
            return self.decoder(z, c)


def cvae_loss(
    x: torch.Tensor,
    x_reconstructed: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CVAE loss: reconstruction loss (MSE) + beta * KL divergence.

    The KL term regularizes the latent space to stay close to N(0, I).
    Beta controls the trade-off: higher beta produces a more organized
    latent space but may hurt reconstruction quality.

    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_loss).
    """
    reconstruction_loss = F.mse_loss(x_reconstructed, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = reconstruction_loss + beta * kl_loss
    return total_loss, reconstruction_loss, kl_loss
