from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model_utils import (
    PROFILE_DIM,
    compute_reconstruction_loss,
    evaluate_generated_scenarios,
    get_default_results_dir,
    get_default_splits_dir,
    load_calendar,
    load_dates,
    load_x_scaler,
    load_x_splits,
    print_split_checks,
    save_extended_metrics,
    save_generation_plots,
    save_interpretation,
    save_json,
    select_device,
    set_random_seed,
    validate_x_splits,
)


class VAE(nn.Module):
    """Unconditional variational autoencoder for daily demand profiles."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode profiles into latent mean and log-variance."""
        hidden = self.encoder(x)
        return self.mu_layer(hidden), self.logvar_layer(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample latent vectors using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors into scaled profiles."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct profiles and return latent distribution parameters."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar


def make_loader(X: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Create a dataloader for profile-only VAE training."""
    tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def vae_loss(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute VAE total loss, reconstruction MSE and KL divergence."""
    reconstruction_loss = F.mse_loss(reconstructed, target, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = reconstruction_loss + beta * kl_loss
    return total_loss, reconstruction_loss, kl_loss


def beta_for_epoch(beta: float, epoch: int, warmup_epochs: int) -> float:
    """Linearly warm up beta and keep it fixed after warm-up."""
    if warmup_epochs <= 0:
        return beta
    return beta * min(1.0, epoch / warmup_epochs)


def run_vae_epoch(
    model: VAE,
    loader: DataLoader,
    device: torch.device,
    beta: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    """Run one VAE epoch and return mean total, reconstruction and KL losses."""
    training = optimizer is not None
    model.train(training)
    total_losses = []
    reconstruction_losses = []
    kl_losses = []

    for (x_batch,) in loader:
        x_batch = x_batch.to(device)
        if training:
            optimizer.zero_grad()
        reconstructed, mu, logvar = model(x_batch)
        total_loss, reconstruction_loss, kl_loss = vae_loss(
            reconstructed,
            x_batch,
            mu,
            logvar,
            beta,
        )
        if training:
            total_loss.backward()
            optimizer.step()

        total_losses.append(total_loss.item())
        reconstruction_losses.append(reconstruction_loss.item())
        kl_losses.append(kl_loss.item())

    return float(np.mean(total_losses)), float(np.mean(reconstruction_losses)), float(np.mean(kl_losses))


def train_vae(
    model: VAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    beta: float,
    beta_warmup_epochs: int,
    patience: int,
    min_delta: float,
) -> dict[str, Any]:
    """Train the unconditional VAE with validation early stopping."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = {
        "epoch": [],
        "beta": [],
        "train_total": [],
        "train_reconstruction": [],
        "train_kl": [],
        "val_total": [],
        "val_reconstruction": [],
        "val_kl": [],
    }
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        current_beta = beta_for_epoch(beta, epoch, beta_warmup_epochs)
        train_total, train_reconstruction, train_kl = run_vae_epoch(
            model,
            train_loader,
            device,
            current_beta,
            optimizer,
        )
        with torch.no_grad():
            val_total, val_reconstruction, val_kl = run_vae_epoch(
                model,
                val_loader,
                device,
                beta,
            )

        history["epoch"].append(epoch)
        history["beta"].append(current_beta)
        history["train_total"].append(train_total)
        history["train_reconstruction"].append(train_reconstruction)
        history["train_kl"].append(train_kl)
        history["val_total"].append(val_total)
        history["val_reconstruction"].append(val_reconstruction)
        history["val_kl"].append(val_kl)

        improved = val_total < best_val_loss - min_delta
        if improved:
            best_val_loss = val_total
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | beta={current_beta:.4f} | "
                f"train_total={train_total:.6f} | val_total={val_total:.6f}"
            )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = float(best_val_loss)
    return history


def reconstruct_array(model: VAE, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    """Reconstruct scaled profiles with the trained VAE mean encoder path."""
    model.eval()
    loader = make_loader(X, batch_size=batch_size, shuffle=False)
    batches = []

    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            mu, _ = model.encode(x_batch)
            reconstructed = model.decode(mu)
            batches.append(reconstructed.cpu().numpy())

    return np.vstack(batches).astype(np.float32)


def generate_vae_scenarios(
    model: VAE,
    n_days: int,
    n_scenarios: int,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Generate unconditional VAE scenarios from the standard normal prior."""
    model.eval()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    total_samples = n_days * n_scenarios
    batches = []

    with torch.no_grad():
        for start in range(0, total_samples, batch_size):
            current_size = min(batch_size, total_samples - start)
            z = torch.randn(current_size, model.latent_dim, device=device)
            decoded = model.decode(z)
            batches.append(decoded.cpu().numpy())

    generated = np.vstack(batches).astype(np.float32)
    return generated.reshape(n_days, n_scenarios, PROFILE_DIM)


def load_vae_checkpoint(path: Path, device: torch.device) -> tuple[VAE, dict[str, Any]]:
    """Load a saved VAE checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model = VAE(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        latent_dim=int(checkpoint["latent_dim"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def encode_distribution_array(
    model: VAE,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a full array into latent means and log-variances."""
    model.eval()
    loader = make_loader(X, batch_size=batch_size, shuffle=False)
    mu_batches = []
    logvar_batches = []

    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            mu, logvar = model.encode(x_batch)
            mu_batches.append(mu.cpu().numpy())
            logvar_batches.append(logvar.cpu().numpy())

    return np.vstack(mu_batches).astype(np.float32), np.vstack(logvar_batches).astype(np.float32)


def latent_kl_by_dimension(mu: np.ndarray, logvar: np.ndarray) -> pd.DataFrame:
    """Compute latent posterior diagnostics by dimension."""
    std = np.exp(0.5 * logvar)
    kl = -0.5 * np.mean(1.0 + logvar - mu**2 - np.exp(logvar), axis=0)
    rows = []

    for index in range(mu.shape[1]):
        rows.append(
            {
                "latent_dim": int(index + 1),
                "mu_mean": float(mu[:, index].mean()),
                "mu_std": float(mu[:, index].std()),
                "logvar_mean": float(logvar[:, index].mean()),
                "std_mean": float(std[:, index].mean()),
                "kl": float(kl[index]),
            }
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the unconditional VAE."""
    parser = argparse.ArgumentParser(description="Train VAE without conditioning from split files.")
    parser.add_argument("--splits-dir", type=Path, default=get_default_splits_dir())
    parser.add_argument("--output-dir", type=Path, default=get_default_results_dir("vae_without_conditioning"))
    parser.add_argument("--pretrained-vae-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--beta-warmup-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--n-scenarios", type=int, default=500)
    parser.add_argument("--mmd-max-samples", type=int, default=1000)
    parser.add_argument("--plot-days", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    """Parse command-line arguments and run the unconditional VAE."""
    args = parse_args()
    run_vae_without_conditioning(args)


def run_vae_without_conditioning(args: argparse.Namespace) -> dict[str, Path]:
    """Run the unconditional VAE end to end and return output paths."""
    set_random_seed(args.seed)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading model inputs from: {args.splits_dir}")
    X = load_x_splits(args.splits_dir)
    validate_x_splits(X)
    print_split_checks(X)

    calendar_test = load_calendar(args.splits_dir, "test")
    dates_test = load_dates(args.splits_dir, "test")
    x_scaler = load_x_scaler(args.splits_dir)
    pretrained_path = getattr(args, "pretrained_vae_path", None)
    history_output_path = args.output_dir / "training_history.csv"
    history_source_path = None
    if pretrained_path is not None and Path(pretrained_path).exists():
        model, checkpoint = load_vae_checkpoint(Path(pretrained_path), device)
        if history_output_path.exists() and len(pd.read_csv(history_output_path)) > 1:
            history_source_path = history_output_path
        history = {
            "epoch": [int(checkpoint.get("best_epoch", 0))],
            "beta": [float(checkpoint.get("beta", args.beta))],
            "train_total": [np.nan],
            "train_reconstruction": [np.nan],
            "train_kl": [np.nan],
            "val_total": [float(checkpoint.get("best_val_loss", np.nan))],
            "val_reconstruction": [np.nan],
            "val_kl": [np.nan],
            "best_epoch": int(checkpoint.get("best_epoch", 0)),
            "best_val_loss": float(checkpoint.get("best_val_loss", np.nan)),
        }
        print(f"Loaded pretrained VAE from: {pretrained_path}")
    else:
        train_loader = make_loader(X["train"], args.batch_size, shuffle=True)
        val_loader = make_loader(X["val"], args.batch_size, shuffle=False)
        model = VAE(PROFILE_DIM, args.hidden_dim, args.latent_dim)
        history = train_vae(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            beta=args.beta,
            beta_warmup_epochs=args.beta_warmup_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
        )

    generated_test = generate_vae_scenarios(
        model=model,
        n_days=len(X["test"]),
        n_scenarios=args.n_scenarios,
        seed=args.seed,
        device=device,
        batch_size=args.batch_size,
    )
    overall_metrics, province_metrics = evaluate_generated_scenarios(
        real_scaled=X["test"],
        generated_scaled=generated_test,
        scaler=x_scaler,
        seed=args.seed,
        mmd_max_samples=args.mmd_max_samples,
    )

    reconstruction_rows = []
    for split_name, values in X.items():
        reconstructed = reconstruct_array(model, values, device, args.batch_size)
        reconstruction_rows.append(
            {"split": split_name, "reconstruction_mse_scaled": compute_reconstruction_loss(reconstructed, values)}
        )

    mu_test, logvar_test = encode_distribution_array(
        model=model,
        X=X["test"],
        device=device,
        batch_size=args.batch_size,
    )
    latent_kl = latent_kl_by_dimension(mu_test, logvar_test)
    best_epoch_value = history["best_epoch"]
    best_val_loss_value = history["best_val_loss"]
    if isinstance(best_epoch_value, list):
        best_epoch_value = best_epoch_value[-1]
    if isinstance(best_val_loss_value, list):
        best_val_loss_value = best_val_loss_value[-1]

    checkpoint_output_path = args.output_dir / "vae_best.pt"
    same_checkpoint = (
        pretrained_path is not None
        and Path(pretrained_path).exists()
        and Path(pretrained_path).resolve() == checkpoint_output_path.resolve()
    )
    if not same_checkpoint:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": PROFILE_DIM,
                "hidden_dim": args.hidden_dim,
                "latent_dim": args.latent_dim,
                "beta": args.beta,
                "best_epoch": int(best_epoch_value),
                "best_val_loss": float(best_val_loss_value),
            },
            checkpoint_output_path,
        )

    if history_source_path is not None:
        pd.read_csv(history_source_path).to_csv(history_output_path, index=False)
    else:
        pd.DataFrame(history).to_csv(history_output_path, index=False)
    pd.DataFrame(reconstruction_rows).to_csv(args.output_dir / "reconstruction_metrics.csv", index=False)
    overall_metrics.to_csv(args.output_dir / "metrics_overall.csv", index=False)
    province_metrics.to_csv(args.output_dir / "metrics_by_province.csv", index=False)
    latent_kl.to_csv(args.output_dir / "latent_kl_by_dimension.csv", index=False)
    save_extended_metrics(
        real_scaled=X["test"],
        generated_scaled=generated_test,
        scaler=x_scaler,
        calendar_test=calendar_test,
        output_dir=args.output_dir,
        seed=args.seed,
        mmd_max_samples=args.mmd_max_samples,
    )
    save_generation_plots(
        real_scaled=X["test"],
        generated_scaled=generated_test,
        scaler=x_scaler,
        dates=dates_test,
        output_dir=args.output_dir / "plots",
        n_days=args.plot_days,
    )
    save_json(
        {
            "model": "vae_without_conditioning",
            "data_source": str(args.splits_dir),
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "beta": args.beta,
            "beta_warmup_epochs": args.beta_warmup_epochs,
            "patience": args.patience,
            "n_scenarios": args.n_scenarios,
            "seed": args.seed,
            "device": str(device),
        },
        args.output_dir / "config.json",
    )
    save_interpretation(
        """
        This VAE learns the marginal distribution of daily profiles X only.
        Generated scenarios are sampled from the same unconditional prior for every
        test day, so they are not adapted to weather, calendar variables or province
        labels.
        """,
        args.output_dir / "interpretation.txt",
    )
    print(f"Saved unconditional VAE outputs to: {args.output_dir}")
    return {
        "output_dir": args.output_dir,
        "overall_metrics": args.output_dir / "metrics_overall.csv",
        "province_metrics": args.output_dir / "metrics_by_province.csv",
        "model": args.output_dir / "vae_best.pt",
    }


if __name__ == "__main__":
    main()
