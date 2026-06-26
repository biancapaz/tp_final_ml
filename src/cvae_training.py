from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "tp_final_ml_matplotlib_cache"),
)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.cvae import CVAE, cvae_loss

HOURS = list(range(1, 25))


def set_random_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch seeds for reproducible CVAE runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def beta_for_epoch(beta: float, epoch: int, warmup_epochs: int) -> float:
    """Linearly increase beta during warm-up, then keep it fixed."""
    if warmup_epochs <= 0:
        return beta
    return beta * min(1.0, epoch / warmup_epochs)


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
    val_loader: DataLoader,
    n_epochs: int = 200,
    learning_rate: float = 1e-3,
    beta: float = 0.01,
    beta_warmup_epochs: int = 50,
    patience: int = 30,
    min_delta: float = 1e-5,
    weight_decay: float = 1e-5,
    device: torch.device = torch.device("cpu"),
    verbose_every: int = 20,
) -> dict:
    """
    Full CVAE training loop with validation, early stopping and beta warm-up.

    The best checkpoint is selected by validation total loss computed with the
    final beta, not by test reconstruction. Test remains untouched for final
    evaluation.
    """
    model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = {
        "epoch": [],
        "beta": [],
        "train_total": [], "train_recon": [], "train_kl": [],
        "val_total": [], "val_recon": [], "val_kl": [],
    }

    best_val_total = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_epoch = n_epochs

    for epoch in range(1, n_epochs + 1):
        current_beta = beta_for_epoch(beta, epoch, beta_warmup_epochs)
        train_total, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, current_beta, device
        )
        val_total, val_recon, val_kl = eval_epoch(
            model, val_loader, beta, device
        )

        history["epoch"].append(epoch)
        history["beta"].append(current_beta)
        history["train_total"].append(train_total)
        history["train_recon"].append(train_recon)
        history["train_kl"].append(train_kl)
        history["val_total"].append(val_total)
        history["val_recon"].append(val_recon)
        history["val_kl"].append(val_kl)

        improved = val_total < best_val_total - min_delta
        if improved:
            best_val_total = val_total
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose_every and epoch % verbose_every == 0:
            print(
                f"Epoch {epoch:03d} | beta {current_beta:.4f} | "
                f"train {train_total:.4f} (recon {train_recon:.4f}, kl {train_kl:.4f}) | "
                f"val {val_total:.4f} (recon {val_recon:.4f}, kl {val_kl:.4f})"
            )

        if epochs_without_improvement >= patience:
            stopped_epoch = epoch
            print(
                f"Early stopping en epoch {epoch} "
                f"(mejor val_total {best_val_total:.4f} en epoch {best_epoch})"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"Mejor modelo restaurado "
            f"(epoch {best_epoch}, val_total {best_val_total:.4f})"
        )

    history["best_epoch"] = best_epoch
    history["stopped_epoch"] = stopped_epoch
    history["best_val_total"] = best_val_total
    return history


def plot_loss_curves(history: dict) -> None:
    """
    Plot train/validation loss curves: total, reconstruction, KL and beta.

    Useful to diagnose underfitting, overfitting and KL collapse.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    pairs = [
        ("train_total", "val_total", "Loss total"),
        ("train_recon", "val_recon", "Reconstruction loss"),
        ("train_kl", "val_kl", "KL divergence"),
    ]

    epochs = history["epoch"]
    for ax, (train_key, val_key, title) in zip(axes[:3], pairs):
        ax.plot(epochs, history[train_key], label="train", color="#4878CF")
        ax.plot(epochs, history[val_key], label="val", color="#D65F5F")
        ax.set_title(title)
        ax.set_xlabel("Epoca")
        ax.legend()
        ax.grid(linewidth=0.4, alpha=0.5)

    axes[3].plot(epochs, history["beta"], color="#6A994E")
    axes[3].set_title("Beta schedule")
    axes[3].set_xlabel("Epoca")
    axes[3].grid(linewidth=0.4, alpha=0.5)

    fig.suptitle("Curvas de entrenamiento del CVAE")
    fig.tight_layout()
    plt.show()


def plot_reconstruction_vs_real(
    model: CVAE,
    X: np.ndarray,
    C: np.ndarray,
    scaler_stats: dict,
    n_examples: int = 4,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    Plot real vs reconstructed profiles for a random sample of days.

    Reconstruction quality shows whether the model learned the basic profile
    shape before evaluating generation quality.
    """
    model.eval()
    n_examples = min(n_examples, len(X))
    indices = np.random.choice(len(X), n_examples, replace=False)

    X_sample = torch.tensor(X[indices]).to(device)
    C_sample = torch.tensor(C[indices]).to(device)

    with torch.no_grad():
        x_recon, _, _ = model(X_sample, C_sample)

    x_mean, x_std = scaler_stats["mean"], scaler_stats["std"]
    X_real = X[indices] * x_std + x_mean
    X_recon = x_recon.cpu().numpy() * x_std + x_mean

    fig, axes = plt.subplots(1, n_examples, figsize=(14, 4), sharey=True)
    if n_examples == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(HOURS, X_real[i] / 1e3, label="real", color="#4878CF", linewidth=1.8)
        ax.plot(
            HOURS,
            X_recon[i] / 1e3,
            label="reconstruido",
            color="#D65F5F",
            linewidth=1.8,
            linestyle="--",
        )
        ax.set_title(f"Ejemplo {i + 1}")
        ax.set_xlabel("Hora")
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.grid(linewidth=0.4, alpha=0.5)
        if i == 0:
            ax.set_ylabel("Demanda (GWh)")
            ax.legend()

    fig.suptitle("Perfiles reales vs reconstruidos")
    fig.tight_layout()
    plt.show()


def plot_generated_vs_real_by_condition(
    model: CVAE,
    X: np.ndarray,
    C: np.ndarray,
    conditions: object,
    scaler_stats: dict,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    Compare mean generated profiles against mean real profiles for each day type.

    For each day type, generates the same number of profiles as real days and
    compares the average curves.
    """
    model.eval()
    x_mean, x_std = scaler_stats["mean"], scaler_stats["std"]

    day_types = ["habil", "sabado", "domingo", "feriado"]
    colors = {"real": "#4878CF", "generado": "#D65F5F"}

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)

    for ax, day_type in zip(axes, day_types):
        mask = conditions["tipo_dia"].values == day_type
        if mask.sum() == 0:
            ax.set_title(f"{day_type}\n(sin datos)")
            continue

        X_real_group = X[mask] * x_std + x_mean
        C_group = torch.tensor(C[mask]).to(device)

        with torch.no_grad():
            z = torch.randn(mask.sum(), model.latent_dim, device=device)
            X_gen = model.decoder(z, C_group).cpu().numpy() * x_std + x_mean

        mean_real = X_real_group.mean(axis=0) / 1e3
        mean_gen = X_gen.mean(axis=0) / 1e3

        p25_real = np.percentile(X_real_group, 25, axis=0) / 1e3
        p75_real = np.percentile(X_real_group, 75, axis=0) / 1e3

        ax.fill_between(HOURS, p25_real, p75_real, alpha=0.2, color=colors["real"])
        ax.plot(HOURS, mean_real, color=colors["real"], linewidth=1.8, label="real")
        ax.plot(
            HOURS,
            mean_gen,
            color=colors["generado"],
            linewidth=1.8,
            linestyle="--",
            label="generado",
        )

        ax.set_title(f"{day_type}\n(n={mask.sum()})")
        ax.set_xlabel("Hora")
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.grid(linewidth=0.4, alpha=0.5)
        if ax == axes[0]:
            ax.set_ylabel("Demanda (GWh)")
            ax.legend(fontsize=8)

    fig.suptitle("Perfil medio generado vs real por tipo de dia")
    fig.tight_layout()
    plt.show()
