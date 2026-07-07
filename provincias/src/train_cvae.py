"""
train_cvae.py

Training loop for the CVAE with early stopping, beta warm-up,
seed fixing and checkpoint saving.
Works for both Architecture A and B.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from cvae import CVAE, cvae_loss


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def make_dataloader(
    X: np.ndarray,
    C: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Build a DataLoader from numpy arrays."""
    dataset = TensorDataset(
        torch.FloatTensor(X),
        torch.FloatTensor(C),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train(
    model: CVAE,
    X_train: np.ndarray,
    C_train: np.ndarray,
    X_val: np.ndarray,
    C_val: np.ndarray,
    config: dict,
    save_path: Path | None = None,
) -> dict:
    """
    Train the CVAE with early stopping and beta warm-up.

    config keys:
      lr:           learning rate (default 1e-3)
      batch_size:   batch size (default 256)
      max_epochs:   maximum epochs (default 300)
      patience:     early stopping patience (default 30)
      beta:         final KL weight (default 1.0)
      warmup_epochs: epochs to linearly ramp beta from 0 to beta (default 50)
      free_bits:    minimum KL per dimension (default 0.0)
      seed:         random seed (default 42)
      device:       'cuda' or 'cpu' (auto-detected if not specified)

    Returns a dict with training history and best val loss.
    """
    seed         = config.get("seed", 42)
    lr           = config.get("lr", 1e-3)
    batch_size   = config.get("batch_size", 256)
    max_epochs   = config.get("max_epochs", 300)
    patience     = config.get("patience", 30)
    beta_final   = config.get("beta", 1.0)
    warmup_epochs = config.get("warmup_epochs", 50)
    free_bits    = config.get("free_bits", 0.0)
    device_str   = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)
    device = torch.device(device_str)
    model  = model.to(device)

    optimizer   = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-5
    )
    train_loader = make_dataloader(X_train, C_train, batch_size, shuffle=True)
    val_loader   = make_dataloader(X_val,   C_val,   batch_size, shuffle=False)

    history = {
        "train_loss": [], "val_loss": [],
        "train_recon": [], "val_recon": [],
        "train_kl": [],   "val_kl": [],
        "beta": [],
    }
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    print(f"Training on {device_str} | x_dim={model.x_dim} | c_dim={model.c_dim} | "
          f"latent_dim={model.latent_dim}")
    print(f"Train: {len(X_train)} samples | Val: {len(X_val)} samples")
    print("-" * 60)

    for epoch in range(1, max_epochs + 1):
        # Beta warm-up
        if warmup_epochs > 0:
            beta = beta_final * min(1.0, epoch / warmup_epochs)
        else:
            beta = beta_final

        # Train
        model.train()
        train_totals = {"loss": 0, "recon": 0, "kl": 0}
        for x_batch, c_batch in train_loader:
            x_batch = x_batch.to(device)
            c_batch = c_batch.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x_batch, c_batch)
            loss, recon, kl = cvae_loss(x_batch, x_recon, mu, logvar, beta, free_bits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_totals["loss"]  += loss.item()
            train_totals["recon"] += recon.item()
            train_totals["kl"]    += kl.item()

        n_batches_train = len(train_loader)

        # Validate
        model.eval()
        val_totals = {"loss": 0, "recon": 0, "kl": 0}
        with torch.no_grad():
            for x_batch, c_batch in val_loader:
                x_batch = x_batch.to(device)
                c_batch = c_batch.to(device)
                x_recon, mu, logvar = model(x_batch, c_batch)
                loss, recon, kl = cvae_loss(x_batch, x_recon, mu, logvar, beta, free_bits)
                val_totals["loss"]  += loss.item()
                val_totals["recon"] += recon.item()
                val_totals["kl"]    += kl.item()

        n_batches_val = len(val_loader)

        tl = train_totals["loss"]  / n_batches_train
        vl = val_totals["loss"]    / n_batches_val
        tr = train_totals["recon"] / n_batches_train
        vr = val_totals["recon"]   / n_batches_val
        tk = train_totals["kl"]    / n_batches_train
        vk = val_totals["kl"]      / n_batches_val

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_recon"].append(tr)
        history["val_recon"].append(vr)
        history["train_kl"].append(tk)
        history["val_kl"].append(vk)
        history["beta"].append(beta)

        scheduler.step(vl)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | beta={beta:.3f} | "
                  f"train={tl:.4f} (r={tr:.4f} kl={tk:.4f}) | "
                  f"val={vl:.4f} (r={vr:.4f} kl={vk:.4f})")

        # Early stopping — skip comparisons during beta warm-up. val_loss
        # includes beta * kl, and kl typically settles at the free_bits
        # floor early on, so while beta is still ramping up, val_loss rises
        # for purely mechanical reasons. Comparing checkpoints before beta
        # reaches its final value would almost always crown an early,
        # under-trained epoch as "best" regardless of actual model quality.
        if epoch <= warmup_epochs:
            continue

        if vl < best_val_loss:
            best_val_loss = vl
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (best val loss: {best_val_loss:.4f})")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "x_dim": model.x_dim,
            "c_dim": model.c_dim,
            "latent_dim": model.latent_dim,
            "best_val_loss": best_val_loss,
            "history": history,
        }, save_path)
        print(f"Checkpoint guardado en {save_path}")

    history["best_val_loss"] = best_val_loss
    return history


def load_checkpoint(path: Path, model: CVAE) -> dict:
    """Load a saved checkpoint into a model."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
