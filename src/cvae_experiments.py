from __future__ import annotations

import copy

import pandas as pd
import torch

from src.cvae import CVAE
from src.cvae_evaluation import evaluate_model
from src.cvae_preprocessing import build_dataloaders
from src.cvae_training import eval_epoch, set_random_seed, train_cvae


DEFAULT_HYPERPARAMS = {
    "profile_dim": 24,
    "hidden_dim": 128,
    "latent_dim": 8,
    "batch_size": 64,
    "n_epochs": 300,
    "learning_rate": 1e-3,
    "beta": 0.01,
    "beta_warmup_epochs": 50,
    "patience": 35,
    "min_delta": 1e-5,
    "weight_decay": 1e-5,
}


def train_one_seed(
    prepared_data: dict[str, dict],
    preprocessing: dict,
    seed: int,
    hyperparams: dict | None = None,
    device: torch.device = torch.device("cpu"),
    verbose_every: int = 50,
) -> dict:
    """Train and evaluate one CVAE baseline run for a single seed."""
    params = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
    set_random_seed(seed)

    train_loader, val_loader, test_loader = build_dataloaders(
        prepared_data["train"]["X"],
        prepared_data["train"]["C"],
        prepared_data["val"]["X"],
        prepared_data["val"]["C"],
        prepared_data["test"]["X"],
        prepared_data["test"]["C"],
        batch_size=params["batch_size"],
    )

    model = CVAE(
        profile_dim=params["profile_dim"],
        condition_dim=prepared_data["train"]["C"].shape[1],
        hidden_dim=params["hidden_dim"],
        latent_dim=params["latent_dim"],
    )

    history = train_cvae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=params["n_epochs"],
        learning_rate=params["learning_rate"],
        beta=params["beta"],
        beta_warmup_epochs=params["beta_warmup_epochs"],
        patience=params["patience"],
        min_delta=params["min_delta"],
        weight_decay=params["weight_decay"],
        device=device,
        verbose_every=verbose_every,
    )

    final_beta = params["beta"]
    train_loss = eval_epoch(model, train_loader, final_beta, device)
    val_loss = eval_epoch(model, val_loader, final_beta, device)
    test_loss = eval_epoch(model, test_loader, final_beta, device)

    loss_summary = pd.DataFrame(
        [
            {
                "seed": seed,
                "split": "train",
                "total_loss": train_loss[0],
                "recon_loss": train_loss[1],
                "kl_loss": train_loss[2],
            },
            {
                "seed": seed,
                "split": "val",
                "total_loss": val_loss[0],
                "recon_loss": val_loss[1],
                "kl_loss": val_loss[2],
            },
            {
                "seed": seed,
                "split": "test",
                "total_loss": test_loss[0],
                "recon_loss": test_loss[1],
                "kl_loss": test_loss[2],
            },
        ]
    )

    metric_frames = []
    for split_name in ["val", "test"]:
        split_metrics = evaluate_model(
            model=model,
            split_name=split_name,
            split_data=prepared_data[split_name],
            scaler_stats=preprocessing["x_scaler"],
            device=device,
            generation_seed=seed,
        )
        split_metrics.insert(0, "seed", seed)
        metric_frames.append(split_metrics)

    metrics = pd.concat(metric_frames, ignore_index=True)
    run_config = {
        **copy.deepcopy(params),
        "seed": seed,
        "condition_dim": prepared_data["train"]["C"].shape[1],
        "condition_encoding": preprocessing["condition_metadata"]["encoding"],
        "best_epoch": history["best_epoch"],
        "stopped_epoch": history["stopped_epoch"],
        "best_val_total": history["best_val_total"],
    }

    return {
        "seed": seed,
        "model": model,
        "history": history,
        "loss_summary": loss_summary,
        "metrics": metrics,
        "config": run_config,
    }


def run_seed_experiments(
    prepared_data: dict[str, dict],
    preprocessing: dict,
    seeds: list[int],
    hyperparams: dict | None = None,
    device: torch.device = torch.device("cpu"),
    verbose_every: int = 50,
) -> dict:
    """Train the final baseline across multiple seeds."""
    runs = []
    loss_frames = []
    metric_frames = []
    config_rows = []

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        run = train_one_seed(
            prepared_data=prepared_data,
            preprocessing=preprocessing,
            seed=seed,
            hyperparams=hyperparams,
            device=device,
            verbose_every=verbose_every,
        )
        runs.append(run)
        loss_frames.append(run["loss_summary"])
        metric_frames.append(run["metrics"])
        config_rows.append(run["config"])

    losses = pd.concat(loss_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    configs = pd.DataFrame(config_rows)
    best_idx = configs["best_val_total"].idxmin()

    return {
        "runs": runs,
        "losses": losses,
        "metrics": metrics,
        "configs": configs,
        "best_run": runs[int(best_idx)],
    }

