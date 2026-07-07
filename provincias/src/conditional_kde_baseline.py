from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import KernelDensity
from torch.utils.data import DataLoader, TensorDataset

from model_utils import (
    PROFILE_DIM,
    compute_reconstruction_loss,
    evaluate_generated_scenarios,
    evaluate_overall_generated_scenarios,
    fallback_level_counts,
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


class Autoencoder(nn.Module):
    """Simple autoencoder used to train KDEs in a compact latent space."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct a scaled demand profile."""
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a scaled demand profile into the latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors into scaled demand profiles."""
        return self.decoder(z)


def make_loader(X: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Create a dataloader for profile-only training."""
    tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_autoencoder_epoch(
    model: Autoencoder,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Run one autoencoder epoch and return mean MSE loss."""
    training = optimizer is not None
    model.train(training)
    losses = []
    criterion = nn.MSELoss()

    for (x_batch,) in loader:
        x_batch = x_batch.to(device)
        if training:
            optimizer.zero_grad()
        reconstructed = model(x_batch)
        loss = criterion(reconstructed, x_batch)
        if training:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses))


def train_autoencoder(
    model: Autoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    min_delta: float,
) -> dict[str, Any]:
    """Train the autoencoder with early stopping on validation MSE."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = {"epoch": [], "train_loss": [], "val_loss": []}
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = run_autoencoder_epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            val_loss = run_autoencoder_epoch(model, val_loader, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val_loss - min_delta
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | train_mse={train_loss:.6f} | val_mse={val_loss:.6f}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = float(best_val_loss)
    return history


def encode_array(model: Autoencoder, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    """Encode a full array into autoencoder latent vectors."""
    model.eval()
    loader = make_loader(X, batch_size=batch_size, shuffle=False)
    batches = []

    with torch.no_grad():
        for (x_batch,) in loader:
            z_batch = model.encode(x_batch.to(device))
            batches.append(z_batch.cpu().numpy())

    return np.vstack(batches).astype(np.float32)


def decode_array(model: Autoencoder, Z: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    """Decode latent vectors into scaled profiles."""
    model.eval()
    batches = []

    with torch.no_grad():
        for start in range(0, len(Z), batch_size):
            z_batch = torch.tensor(Z[start : start + batch_size], dtype=torch.float32, device=device)
            x_batch = model.decode(z_batch)
            batches.append(x_batch.cpu().numpy())

    return np.vstack(batches).astype(np.float32)


def reconstruct_array(
    model: Autoencoder,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Reconstruct scaled profiles with the trained autoencoder."""
    z = encode_array(model, X, device, batch_size)
    return decode_array(model, z, device, batch_size)


def calendar_key(row: pd.Series, level: str) -> tuple[int, ...]:
    """Build a hierarchical calendar key for one row."""
    if level == "full":
        return int(row["mes"]), int(row["dia_semana"]), int(row["es_feriado"])
    if level == "month_weekday":
        return int(row["mes"]), int(row["dia_semana"])
    if level == "month":
        return (int(row["mes"]),)
    raise ValueError(f"Unknown calendar level: {level}")


def group_indices(calendar: pd.DataFrame, level: str) -> dict[tuple[int, ...], np.ndarray]:
    """Group row indices by a calendar key."""
    keys = calendar.apply(lambda row: calendar_key(row, level), axis=1)
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    return {key: np.array(indices, dtype=int) for key, indices in groups.items()}


def build_calendar_group_counts(calendar: pd.DataFrame) -> pd.DataFrame:
    """Create a table with training sample counts for each calendar fallback level."""
    rows = []
    for level in ["full", "month_weekday", "month"]:
        groups = group_indices(calendar, level)
        for key, indices in groups.items():
            rows.append({"level": level, "key": "|".join(map(str, key)), "n_samples": len(indices)})
    rows.append({"level": "global", "key": "global", "n_samples": len(calendar)})
    return pd.DataFrame(rows)


def fit_kdes(
    Z_train: np.ndarray,
    calendar_train: pd.DataFrame,
    bandwidth: float,
    min_samples: int,
) -> dict[str, Any]:
    """Fit KDE models for every calendar fallback level."""
    kdes: dict[str, Any] = {"full": {}, "month_weekday": {}, "month": {}}

    for level in ["full", "month_weekday", "month"]:
        groups = group_indices(calendar_train, level)
        for key, indices in groups.items():
            if len(indices) >= min_samples:
                kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
                kde.fit(Z_train[indices])
                kdes[level][key] = kde

    global_kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    global_kde.fit(Z_train)
    kdes["global"] = global_kde
    return kdes


def select_kde(row: pd.Series, kdes: dict[str, Any]) -> tuple[KernelDensity, str, tuple[int, ...] | str]:
    """Select the most specific available KDE for one calendar row."""
    for level in ["full", "month_weekday", "month"]:
        key = calendar_key(row, level)
        if key in kdes[level]:
            return kdes[level][key], level, key
    return kdes["global"], "global", "global"


def score_bandwidth(
    Z_train: np.ndarray,
    Z_val: np.ndarray,
    calendar_train: pd.DataFrame,
    calendar_val: pd.DataFrame,
    bandwidth: float,
    min_samples: int,
) -> float:
    """Fit hierarchical KDEs and return average validation log-likelihood."""
    kdes = fit_kdes(Z_train, calendar_train, bandwidth, min_samples)
    log_likelihoods = []

    for index, row in calendar_val.iterrows():
        kde, _, _ = select_kde(row, kdes)
        score = kde.score_samples(Z_val[index : index + 1])[0]
        log_likelihoods.append(score)

    return float(np.mean(log_likelihoods))


def select_bandwidth(
    Z_train: np.ndarray,
    Z_val: np.ndarray,
    calendar_train: pd.DataFrame,
    calendar_val: pd.DataFrame,
    bandwidths: list[float],
    min_samples: int,
) -> tuple[float, pd.DataFrame]:
    """Select KDE bandwidth by validation log-likelihood."""
    rows = []
    for bandwidth in bandwidths:
        mean_log_likelihood = score_bandwidth(
            Z_train=Z_train,
            Z_val=Z_val,
            calendar_train=calendar_train,
            calendar_val=calendar_val,
            bandwidth=bandwidth,
            min_samples=min_samples,
        )
        rows.append({"bandwidth": bandwidth, "val_mean_log_likelihood": mean_log_likelihood})
        print(f"Bandwidth={bandwidth:.4f} | val_mean_log_likelihood={mean_log_likelihood:.4f}")

    results = pd.DataFrame(rows).sort_values("val_mean_log_likelihood", ascending=False)
    best_bandwidth = float(results.iloc[0]["bandwidth"])
    print(f"Selected bandwidth={best_bandwidth:.4f}")
    return best_bandwidth, results


def evaluate_kde_validation_configuration(
    model: Autoencoder,
    Z_train: np.ndarray,
    Z_val: np.ndarray,
    X_val: np.ndarray,
    calendar_train: pd.DataFrame,
    calendar_val: pd.DataFrame,
    x_scaler: Any,
    min_samples: int,
    bandwidth: float,
    n_scenarios: int,
    seed: int,
    device: torch.device,
    batch_size: int,
    mmd_max_samples: int,
) -> dict[str, float | int]:
    """Evaluate one hierarchical KDE configuration only on validation."""
    kdes = fit_kdes(Z_train, calendar_train, bandwidth, min_samples)
    log_likelihoods = []

    for index, row in calendar_val.iterrows():
        kde, _, _ = select_kde(row, kdes)
        log_likelihoods.append(kde.score_samples(Z_val[index : index + 1])[0])

    generated_val, fallback_usage = generate_kde_scenarios(
        model=model,
        kdes=kdes,
        calendar_test=calendar_val,
        n_scenarios=n_scenarios,
        seed=seed,
        device=device,
        batch_size=batch_size,
    )
    overall_metrics = evaluate_overall_generated_scenarios(
        real_scaled=X_val,
        generated_scaled=generated_val,
        scaler=x_scaler,
        seed=seed,
        mmd_max_samples=mmd_max_samples,
    )
    overall = overall_metrics.iloc[0]
    row = {
        "min_samples": int(min_samples),
        "bandwidth": float(bandwidth),
        "val_log_likelihood": float(np.mean(log_likelihoods)),
        "val_coverage": float(overall["coverage_p5_p95"]),
        "val_sharpness": float(overall["sharpness_p5_p95_mwh"]),
        "val_rmse_mean_profile": float(overall["mean_profile_rmse_mwh"]),
        "val_mmd": float(overall["mmd_rbf"]),
    }
    row.update(fallback_level_counts(fallback_usage))
    row["coverage_gap"] = abs(row["val_coverage"] - 0.90)
    return row


def run_kde_sensitivity_analysis(
    model: Autoencoder,
    Z_train: np.ndarray,
    Z_val: np.ndarray,
    X_val: np.ndarray,
    calendar_train: pd.DataFrame,
    calendar_val: pd.DataFrame,
    x_scaler: Any,
    min_samples_grid: list[int],
    bandwidth_grid: list[float],
    n_scenarios: int,
    seed: int,
    device: torch.device,
    batch_size: int,
    mmd_max_samples: int,
) -> pd.DataFrame:
    """Run min_samples and bandwidth sensitivity using validation only."""
    rows = []

    for min_samples in min_samples_grid:
        for bandwidth in bandwidth_grid:
            row = evaluate_kde_validation_configuration(
                model=model,
                Z_train=Z_train,
                Z_val=Z_val,
                X_val=X_val,
                calendar_train=calendar_train,
                calendar_val=calendar_val,
                x_scaler=x_scaler,
                min_samples=min_samples,
                bandwidth=bandwidth,
                n_scenarios=n_scenarios,
                seed=seed,
                device=device,
                batch_size=batch_size,
                mmd_max_samples=mmd_max_samples,
            )
            rows.append(row)
            print(
                f"min_samples={min_samples} | bandwidth={bandwidth:.2f} | "
                f"val_ll={row['val_log_likelihood']:.4f} | "
                f"val_coverage={row['val_coverage']:.3f}"
            )

    sensitivity = pd.DataFrame(rows).sort_values("val_log_likelihood", ascending=False)
    if len(sensitivity) == 0:
        raise RuntimeError("KDE sensitivity analysis did not evaluate any configuration")
    return sensitivity


def select_kde_validation_configurations(sensitivity: pd.DataFrame) -> pd.DataFrame:
    """Select KDE configurations by validation log-likelihood and generative metrics."""
    log_likelihood_row = (
        sensitivity.sort_values("val_log_likelihood", ascending=False)
        .iloc[0]
        .to_dict()
    )
    generative_row = (
        sensitivity.sort_values(
            ["coverage_gap", "val_sharpness", "val_rmse_mean_profile"],
            ascending=[True, True, True],
        )
        .iloc[0]
        .to_dict()
    )
    rows = []
    for criterion, row in [
        ("validation_log_likelihood", log_likelihood_row),
        ("validation_generative", generative_row),
    ]:
        selected = {"selection_criterion": criterion}
        selected.update(row)
        rows.append(selected)
    return pd.DataFrame(rows)


def evaluate_kde_selection_on_test(
    model: Autoencoder,
    Z_train: np.ndarray,
    calendar_train: pd.DataFrame,
    calendar_test: pd.DataFrame,
    X_test: np.ndarray,
    x_scaler: Any,
    selection: pd.Series,
    n_scenarios: int,
    seed: int,
    device: torch.device,
    batch_size: int,
    mmd_max_samples: int,
) -> tuple[dict[str, float | int | str], np.ndarray, pd.DataFrame]:
    """Evaluate one already-selected KDE configuration on test."""
    min_samples = int(selection["min_samples"])
    bandwidth = float(selection["bandwidth"])
    kdes = fit_kdes(Z_train, calendar_train, bandwidth, min_samples)
    generated_test, fallback_usage = generate_kde_scenarios(
        model=model,
        kdes=kdes,
        calendar_test=calendar_test,
        n_scenarios=n_scenarios,
        seed=seed,
        device=device,
        batch_size=batch_size,
    )
    overall_metrics = evaluate_overall_generated_scenarios(
        real_scaled=X_test,
        generated_scaled=generated_test,
        scaler=x_scaler,
        seed=seed,
        mmd_max_samples=mmd_max_samples,
    )
    overall = overall_metrics.iloc[0]
    fallback_counts = fallback_level_counts(fallback_usage)
    row = {
        "selection_criterion": str(selection["selection_criterion"]),
        "min_samples": min_samples,
        "bandwidth": bandwidth,
        "val_log_likelihood": float(selection["val_log_likelihood"]),
        "val_coverage": float(selection["val_coverage"]),
        "val_sharpness": float(selection["val_sharpness"]),
        "val_rmse_mean_profile": float(selection["val_rmse_mean_profile"]),
        "val_mmd": float(selection["val_mmd"]),
        "test_coverage": float(overall["coverage_p5_p95"]),
        "test_sharpness": float(overall["sharpness_p5_p95_mwh"]),
        "test_rmse_mean_profile": float(overall["mean_profile_rmse_mwh"]),
        "test_mmd": float(overall["mmd_rbf"]),
        "n_full_test": fallback_counts["n_full"],
        "n_month_weekday_test": fallback_counts["n_month_weekday"],
        "n_month_test": fallback_counts["n_month"],
        "n_global_test": fallback_counts["n_global"],
    }
    return row, generated_test, fallback_usage


def load_autoencoder_checkpoint(path: Path, device: torch.device) -> tuple[Autoencoder, dict[str, Any]]:
    """Load a saved autoencoder checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model = Autoencoder(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        latent_dim=int(checkpoint["latent_dim"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def generate_kde_scenarios(
    model: Autoencoder,
    kdes: dict[str, Any],
    calendar_test: pd.DataFrame,
    n_scenarios: int,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Generate decoded test scenarios using calendar-conditioned KDE fallback."""
    rng = np.random.default_rng(seed)
    latent_samples = []
    usage_rows = []

    for day_index, row in calendar_test.iterrows():
        kde, level, key = select_kde(row, kdes)
        random_state = int(rng.integers(0, np.iinfo(np.int32).max))
        z_samples = kde.sample(n_scenarios, random_state=random_state).astype(np.float32)
        latent_samples.append(z_samples)
        usage_rows.append(
            {
                "day_index": int(day_index),
                "fecha": row["fecha"],
                "fallback_level": level,
                "fallback_key": "|".join(map(str, key)) if isinstance(key, tuple) else key,
            }
        )

    stacked_latent = np.vstack(latent_samples)
    decoded = decode_array(model, stacked_latent, device, batch_size)
    scenarios = decoded.reshape(len(calendar_test), n_scenarios, PROFILE_DIM)
    return scenarios.astype(np.float32), pd.DataFrame(usage_rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the conditional KDE baseline."""
    parser = argparse.ArgumentParser(description="Train conditional KDE baseline from split files.")
    parser.add_argument("--splits-dir", type=Path, default=get_default_splits_dir())
    parser.add_argument("--output-dir", type=Path, default=get_default_results_dir("conditional_kde"))
    parser.add_argument("--pretrained-autoencoder-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--min-samples-grid", type=int, nargs="+", default=[3, 5, 10, 15, 20])
    parser.add_argument(
        "--bandwidths",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    )
    parser.add_argument("--n-scenarios", type=int, default=500)
    parser.add_argument("--mmd-max-samples", type=int, default=1000)
    parser.add_argument("--plot-days", type=int, default=4)
    parser.add_argument(
        "--final-selection-criterion",
        type=str,
        choices=["validation_log_likelihood", "validation_generative"],
        default="validation_generative",
    )
    return parser.parse_args()


def main() -> None:
    """Parse command-line arguments and run the conditional KDE baseline."""
    args = parse_args()
    run_conditional_kde_baseline(args)


def run_conditional_kde_baseline(args: argparse.Namespace) -> dict[str, Path]:
    """Run the conditional KDE baseline end to end and return output paths."""
    set_random_seed(args.seed)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading model inputs from: {args.splits_dir}")
    X = load_x_splits(args.splits_dir)
    validate_x_splits(X)
    print_split_checks(X)

    calendar_train = load_calendar(args.splits_dir, "train")
    calendar_val = load_calendar(args.splits_dir, "val")
    calendar_test = load_calendar(args.splits_dir, "test")
    dates_test = load_dates(args.splits_dir, "test")
    x_scaler = load_x_scaler(args.splits_dir)

    pretrained_path = getattr(args, "pretrained_autoencoder_path", None)
    history_source_path = None
    if pretrained_path is not None and Path(pretrained_path).exists():
        model, checkpoint = load_autoencoder_checkpoint(Path(pretrained_path), device)
        candidate_history_path = Path(pretrained_path).parent / "autoencoder_history.csv"
        if candidate_history_path.exists():
            history_source_path = candidate_history_path
        history = {
            "epoch": [int(checkpoint.get("best_epoch", 0))],
            "train_loss": [np.nan],
            "val_loss": [float(checkpoint.get("best_val_loss", np.nan))],
            "best_epoch": int(checkpoint.get("best_epoch", 0)),
            "best_val_loss": float(checkpoint.get("best_val_loss", np.nan)),
        }
        print(f"Loaded pretrained autoencoder from: {pretrained_path}")
    else:
        train_loader = make_loader(X["train"], args.batch_size, shuffle=True)
        val_loader = make_loader(X["val"], args.batch_size, shuffle=False)
        model = Autoencoder(PROFILE_DIM, args.hidden_dim, args.latent_dim)
        history = train_autoencoder(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            min_delta=args.min_delta,
        )

    Z_train = encode_array(model, X["train"], device, args.batch_size)
    Z_val = encode_array(model, X["val"], device, args.batch_size)

    min_samples_grid = getattr(args, "min_samples_grid", [3, 5, 10, 15, 20])
    sensitivity_results = run_kde_sensitivity_analysis(
        model=model,
        Z_train=Z_train,
        Z_val=Z_val,
        X_val=X["val"],
        calendar_train=calendar_train,
        calendar_val=calendar_val,
        x_scaler=x_scaler,
        min_samples_grid=min_samples_grid,
        bandwidth_grid=args.bandwidths,
        n_scenarios=args.n_scenarios,
        seed=args.seed,
        device=device,
        batch_size=args.batch_size,
        mmd_max_samples=args.mmd_max_samples,
    )
    selected_configs = select_kde_validation_configurations(sensitivity_results)
    comparison_rows = []
    generated_by_criterion: dict[str, np.ndarray] = {}
    fallback_by_criterion: dict[str, pd.DataFrame] = {}

    for _, selection in selected_configs.iterrows():
        comparison_row, generated, fallback = evaluate_kde_selection_on_test(
            model=model,
            Z_train=Z_train,
            calendar_train=calendar_train,
            calendar_test=calendar_test,
            X_test=X["test"],
            x_scaler=x_scaler,
            selection=selection,
            n_scenarios=args.n_scenarios,
            seed=args.seed,
            device=device,
            batch_size=args.batch_size,
            mmd_max_samples=args.mmd_max_samples,
        )
        comparison_rows.append(comparison_row)
        criterion = str(selection["selection_criterion"])
        generated_by_criterion[criterion] = generated
        fallback_by_criterion[criterion] = fallback

    selection_comparison = pd.DataFrame(comparison_rows)
    final_selection_criterion = getattr(args, "final_selection_criterion", "validation_generative")
    if final_selection_criterion not in generated_by_criterion:
        final_selection_criterion = "validation_log_likelihood"
    final_row = selection_comparison[
        selection_comparison["selection_criterion"] == final_selection_criterion
    ].iloc[0]
    generated_test = generated_by_criterion[final_selection_criterion]
    fallback_usage = fallback_by_criterion[final_selection_criterion]
    best_min_samples = int(final_row["min_samples"])
    best_bandwidth = float(final_row["bandwidth"])
    print(
        f"Selected min_samples={best_min_samples} and bandwidth={best_bandwidth:.2f} "
        f"by {final_selection_criterion}"
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

    best_epoch_value = history["best_epoch"]
    best_val_loss_value = history["best_val_loss"]
    if isinstance(best_epoch_value, list):
        best_epoch_value = best_epoch_value[-1]
    if isinstance(best_val_loss_value, list):
        best_val_loss_value = best_val_loss_value[-1]

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": PROFILE_DIM,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "best_epoch": int(best_epoch_value),
            "best_val_loss": float(best_val_loss_value),
        },
        args.output_dir / "autoencoder_best.pt",
    )

    if history_source_path is not None:
        pd.read_csv(history_source_path).to_csv(args.output_dir / "autoencoder_history.csv", index=False)
    else:
        pd.DataFrame(history).to_csv(args.output_dir / "autoencoder_history.csv", index=False)
    pd.DataFrame(reconstruction_rows).to_csv(args.output_dir / "reconstruction_metrics.csv", index=False)
    sensitivity_results.to_csv(
        args.output_dir / "kde_min_samples_bandwidth_sensitivity.csv",
        index=False,
    )
    sensitivity_results.to_csv(args.output_dir / "bandwidth_validation.csv", index=False)
    selection_comparison.to_csv(args.output_dir / "kde_selection_comparison.csv", index=False)
    build_calendar_group_counts(calendar_train).to_csv(args.output_dir / "calendar_group_counts.csv", index=False)
    fallback_usage.to_csv(args.output_dir / "test_fallback_usage.csv", index=False)
    overall_metrics.to_csv(args.output_dir / "metrics_overall.csv", index=False)
    province_metrics.to_csv(args.output_dir / "metrics_by_province.csv", index=False)
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
            "model": "conditional_kde_baseline",
            "data_source": str(args.splits_dir),
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "min_samples_grid": min_samples_grid,
            "bandwidths": args.bandwidths,
            "best_min_samples": best_min_samples,
            "best_bandwidth": best_bandwidth,
            "final_selection_criterion": final_selection_criterion,
            "n_scenarios": args.n_scenarios,
            "seed": args.seed,
            "device": str(device),
        },
        args.output_dir / "config.json",
    )
    save_interpretation(
        """
        This baseline trains an autoencoder on X_train and fits hierarchical KDEs
        on the latent vectors using only discrete calendar groups. It does not use
        weather or socioeconomic conditions.
        """,
        args.output_dir / "interpretation.txt",
    )
    print(f"Saved conditional KDE outputs to: {args.output_dir}")
    return {
        "output_dir": args.output_dir,
        "overall_metrics": args.output_dir / "metrics_overall.csv",
        "province_metrics": args.output_dir / "metrics_by_province.csv",
        "model": args.output_dir / "autoencoder_best.pt",
    }


if __name__ == "__main__":
    main()
