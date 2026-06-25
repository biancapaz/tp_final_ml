import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.cvae import CVAE, cvae_loss

HOURS = list(range(1, 25))


def train_epoch(
    model: CVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    beta: float,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Run one training epoch and return mean losses.

    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_loss).
    """
    model.train()
    total_losses, recon_losses, kl_losses = [], [], []

    for x_batch, c_batch in loader:
        x_batch = x_batch.to(device)
        c_batch = c_batch.to(device)

        optimizer.zero_grad()
        x_reconstructed, mu, log_var = model(x_batch, c_batch)
        loss, recon, kl = cvae_loss(x_batch, x_reconstructed, mu, log_var, beta)
        loss.backward()
        optimizer.step()

        total_losses.append(loss.item())
        recon_losses.append(recon.item())
        kl_losses.append(kl.item())

    return np.mean(total_losses), np.mean(recon_losses), np.mean(kl_losses)


def eval_epoch(
    model: CVAE,
    loader: DataLoader,
    beta: float,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Evaluate the model on a dataloader without updating weights.

    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_loss).
    """
    model.eval()
    total_losses, recon_losses, kl_losses = [], [], []

    with torch.no_grad():
        for x_batch, c_batch in loader:
            x_batch = x_batch.to(device)
            c_batch = c_batch.to(device)
            x_reconstructed, mu, log_var = model(x_batch, c_batch)
            loss, recon, kl = cvae_loss(x_batch, x_reconstructed, mu, log_var, beta)
            total_losses.append(loss.item())
            recon_losses.append(recon.item())
            kl_losses.append(kl.item())

    return np.mean(total_losses), np.mean(recon_losses), np.mean(kl_losses)


def train_cvae(
    model: CVAE,
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_epochs: int = 200,
    learning_rate: float = 1e-3,
    beta: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Full training loop for the CVAE.

    Tracks train and test losses across epochs and returns the history
    for plotting. The model with the lowest test reconstruction loss
    is saved in memory.

    Args:
        beta: Weight of the KL divergence term. Start with 1.0 and
              increase if the latent space is not well organized.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_total": [], "train_recon": [], "train_kl": [],
        "test_total": [], "test_recon": [], "test_kl": [],
    }

    best_test_recon = float("inf")
    best_state = None

    for epoch in range(1, n_epochs + 1):
        train_total, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, beta, device
        )
        test_total, test_recon, test_kl = eval_epoch(
            model, test_loader, beta, device
        )

        history["train_total"].append(train_total)
        history["train_recon"].append(train_recon)
        history["train_kl"].append(train_kl)
        history["test_total"].append(test_total)
        history["test_recon"].append(test_recon)
        history["test_kl"].append(test_kl)

        if test_recon < best_test_recon:
            best_test_recon = test_recon
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss: {train_total:.4f} (recon {train_recon:.4f}, kl {train_kl:.4f}) | "
                f"test loss: {test_total:.4f} (recon {test_recon:.4f})"
            )

    # restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nMejor modelo restaurado (test recon: {best_test_recon:.4f})")

    return history


def plot_loss_curves(history: dict) -> None:
    """
    Plot training and test loss curves: total, reconstruction and KL.

    Useful to diagnose underfitting, overfitting or KL collapse.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    pairs = [
        ("train_total", "test_total", "Loss total"),
        ("train_recon", "test_recon", "Reconstruction loss"),
        ("train_kl", "test_kl", "KL divergence"),
    ]

    for ax, (train_key, test_key, title) in zip(axes, pairs):
        ax.plot(history[train_key], label="train", color="#4878CF")
        ax.plot(history[test_key], label="test", color="#D65F5F")
        ax.set_title(title)
        ax.set_xlabel("Epoca")
        ax.legend()
        ax.grid(linewidth=0.4, alpha=0.5)

    fig.suptitle("Curvas de loss del CVAE")
    fig.tight_layout()
    plt.show()


def plot_reconstruction_vs_real(
    model: CVAE,
    X_test: np.ndarray,
    C_test: np.ndarray,
    scaler_stats: dict,
    n_examples: int = 4,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    Plot real vs reconstructed profiles for a random sample of test days.

    Reconstruction quality shows whether the model learned the basic
    shape of the profiles before evaluating generation quality.
    """
    model.eval()
    indices = np.random.choice(len(X_test), n_examples, replace=False)

    X_sample = torch.tensor(X_test[indices]).to(device)
    C_sample = torch.tensor(C_test[indices]).to(device)

    with torch.no_grad():
        x_recon, _, _ = model(X_sample, C_sample)

    x_mean, x_std = scaler_stats["mean"], scaler_stats["std"]
    X_real = X_test[indices] * x_std + x_mean
    X_recon = x_recon.cpu().numpy() * x_std + x_mean

    fig, axes = plt.subplots(1, n_examples, figsize=(14, 4), sharey=True)

    for i, ax in enumerate(axes):
        ax.plot(HOURS, X_real[i] / 1e3, label="real", color="#4878CF", linewidth=1.8)
        ax.plot(HOURS, X_recon[i] / 1e3, label="reconstruido",
                color="#D65F5F", linewidth=1.8, linestyle="--")
        ax.set_title(f"Ejemplo {i + 1}")
        ax.set_xlabel("Hora")
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.grid(linewidth=0.4, alpha=0.5)
        if i == 0:
            ax.set_ylabel("Demanda (GWh)")
            ax.legend()

    fig.suptitle("Perfiles reales vs reconstruidos (test set)")
    fig.tight_layout()
    plt.show()


def plot_generated_vs_real_by_condition(
    model: CVAE,
    X_test: np.ndarray,
    C_test: np.ndarray,
    dates_test: object,
    conditions_test: object,
    scaler_stats: dict,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    Compare mean generated profiles against mean real profiles for each day type.

    For each day type in the test set, generates the same number of profiles
    as real days available and compares the average curves.
    This is the main evaluation of whether conditioning is working correctly.
    """
    model.eval()
    x_mean, x_std = scaler_stats["mean"], scaler_stats["std"]

    day_types = ["habil", "sabado", "domingo", "feriado"]
    colors = {"real": "#4878CF", "generado": "#D65F5F"}

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)

    for ax, day_type in zip(axes, day_types):
        mask = conditions_test["tipo_dia"].values == day_type
        if mask.sum() == 0:
            ax.set_title(f"{day_type}\n(sin datos)")
            continue

        X_real_group = X_test[mask] * x_std + x_mean
        C_group = torch.tensor(C_test[mask]).to(device)

        with torch.no_grad():
            z = torch.randn(mask.sum(), model.latent_dim, device=device)
            X_gen = model.decoder(z, C_group).cpu().numpy() * x_std + x_mean

        mean_real = X_real_group.mean(axis=0) / 1e3
        mean_gen = X_gen.mean(axis=0) / 1e3

        p25_real = np.percentile(X_real_group, 25, axis=0) / 1e3
        p75_real = np.percentile(X_real_group, 75, axis=0) / 1e3

        ax.fill_between(HOURS, p25_real, p75_real, alpha=0.2, color=colors["real"])
        ax.plot(HOURS, mean_real, color=colors["real"], linewidth=1.8, label="real")
        ax.plot(HOURS, mean_gen, color=colors["generado"],
                linewidth=1.8, linestyle="--", label="generado")

        ax.set_title(f"{day_type}\n(n={mask.sum()})")
        ax.set_xlabel("Hora")
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.grid(linewidth=0.4, alpha=0.5)
        if ax == axes[0]:
            ax.set_ylabel("Demanda (GWh)")
            ax.legend(fontsize=8)

    fig.suptitle("Perfil medio generado vs real por tipo de dia (test set)")
    fig.tight_layout()
    plt.show()
