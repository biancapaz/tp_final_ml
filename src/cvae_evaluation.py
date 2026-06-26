from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.cvae import CVAE
from src.cvae_preprocessing import inverse_transform_profiles

HOURS = np.arange(1, 25)


def _to_original_scale(X_scaled: np.ndarray, scaler_stats: dict) -> np.ndarray:
    return inverse_transform_profiles(X_scaled, scaler_stats)


def reconstruct_scaled(
    model: CVAE,
    X: np.ndarray,
    C: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Return reconstructed standardized profiles."""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        c_tensor = torch.tensor(C, dtype=torch.float32, device=device)
        x_recon, _, _ = model(x_tensor, c_tensor)
    return x_recon.cpu().numpy()


def generate_scaled(
    model: CVAE,
    C: np.ndarray,
    device: torch.device,
    seed: int | None = None,
) -> np.ndarray:
    """Generate standardized profiles from z ~ N(0, I) conditional on C."""
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    with torch.no_grad():
        c_tensor = torch.tensor(C, dtype=torch.float32, device=device)
        z = torch.randn(len(C), model.latent_dim, device=device)
        generated = model.decoder(z, c_tensor)
    return generated.cpu().numpy()


def profile_summary_metrics(
    real_profiles: np.ndarray,
    predicted_profiles: np.ndarray,
) -> dict[str, float]:
    """
    Compare real and predicted/generated profiles in original MWh scale.

    Metrics are meant to be interpretable for electricity demand profiles:
    point-wise error, daily energy, peak demand, peak hour and ramps.
    """
    errors = predicted_profiles - real_profiles
    abs_errors = np.abs(errors)
    squared_errors = errors**2

    real_energy = real_profiles.sum(axis=1)
    pred_energy = predicted_profiles.sum(axis=1)
    real_peak = real_profiles.max(axis=1)
    pred_peak = predicted_profiles.max(axis=1)
    real_peak_hour = real_profiles.argmax(axis=1) + 1
    pred_peak_hour = predicted_profiles.argmax(axis=1) + 1
    real_ramps = np.diff(real_profiles, axis=1)
    pred_ramps = np.diff(predicted_profiles, axis=1)

    return {
        "mae_mwh": float(abs_errors.mean()),
        "rmse_mwh": float(np.sqrt(squared_errors.mean())),
        "energy_mae_mwh": float(np.abs(pred_energy - real_energy).mean()),
        "energy_mape_pct": float(
            np.mean(np.abs(pred_energy - real_energy) / np.maximum(real_energy, 1e-8)) * 100
        ),
        "peak_mae_mwh": float(np.abs(pred_peak - real_peak).mean()),
        "peak_hour_accuracy": float((pred_peak_hour == real_peak_hour).mean()),
        "peak_hour_mae": float(np.abs(pred_peak_hour - real_peak_hour).mean()),
        "ramp_mae_mwh": float(np.abs(pred_ramps - real_ramps).mean()),
    }


def grouped_profile_metrics(
    real_profiles: np.ndarray,
    predicted_profiles: np.ndarray,
    conditions: pd.DataFrame,
) -> dict[str, float]:
    """Compare group mean and percentile profiles by tipo_dia."""
    rows = {}
    group_mean_errors = []
    group_p50_errors = []

    for day_type in ["habil", "sabado", "domingo", "feriado"]:
        mask = conditions["tipo_dia"].to_numpy() == day_type
        if mask.sum() == 0:
            continue

        real_group = real_profiles[mask]
        pred_group = predicted_profiles[mask]
        mean_error = np.abs(pred_group.mean(axis=0) - real_group.mean(axis=0)).mean()
        p50_error = np.abs(
            np.percentile(pred_group, 50, axis=0)
            - np.percentile(real_group, 50, axis=0)
        ).mean()

        rows[f"{day_type}_mean_profile_mae_mwh"] = float(mean_error)
        rows[f"{day_type}_p50_profile_mae_mwh"] = float(p50_error)
        group_mean_errors.append(mean_error)
        group_p50_errors.append(p50_error)

    rows["group_mean_profile_mae_mwh"] = float(np.mean(group_mean_errors))
    rows["group_p50_profile_mae_mwh"] = float(np.mean(group_p50_errors))
    return rows


def diversity_metrics(
    real_profiles: np.ndarray,
    generated_profiles: np.ndarray,
) -> dict[str, float]:
    """
    Compare generated variability against real variability.

    A ratio near 1 means the synthetic profiles have similar dispersion to
    the real profiles. Very low values indicate overly smooth generation.
    """
    real_hourly_std = real_profiles.std(axis=0).mean()
    gen_hourly_std = generated_profiles.std(axis=0).mean()
    return {
        "hourly_std_real_mwh": float(real_hourly_std),
        "hourly_std_generated_mwh": float(gen_hourly_std),
        "hourly_std_ratio_generated_real": float(gen_hourly_std / (real_hourly_std + 1e-8)),
    }


def evaluate_reconstruction(
    model: CVAE,
    X: np.ndarray,
    C: np.ndarray,
    scaler_stats: dict,
    conditions: pd.DataFrame,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate reconstruction on one split."""
    X_recon_scaled = reconstruct_scaled(model, X, C, device)
    real = _to_original_scale(X, scaler_stats)
    recon = _to_original_scale(X_recon_scaled, scaler_stats)
    return {
        **profile_summary_metrics(real, recon),
        **grouped_profile_metrics(real, recon, conditions),
    }


def evaluate_generation(
    model: CVAE,
    X: np.ndarray,
    C: np.ndarray,
    scaler_stats: dict,
    conditions: pd.DataFrame,
    device: torch.device,
    seed: int | None = None,
) -> dict[str, float]:
    """Evaluate one generated sample per condition on one split."""
    X_gen_scaled = generate_scaled(model, C, device, seed=seed)
    real = _to_original_scale(X, scaler_stats)
    generated = _to_original_scale(X_gen_scaled, scaler_stats)
    return {
        **profile_summary_metrics(real, generated),
        **grouped_profile_metrics(real, generated, conditions),
        **diversity_metrics(real, generated),
    }


def evaluate_model(
    model: CVAE,
    split_name: str,
    split_data: dict,
    scaler_stats: dict,
    device: torch.device,
    generation_seed: int | None = None,
) -> pd.DataFrame:
    """Return reconstruction and generation metrics for one split."""
    recon_metrics = evaluate_reconstruction(
        model=model,
        X=split_data["X"],
        C=split_data["C"],
        scaler_stats=scaler_stats,
        conditions=split_data["conditions"],
        device=device,
    )
    gen_metrics = evaluate_generation(
        model=model,
        X=split_data["X"],
        C=split_data["C"],
        scaler_stats=scaler_stats,
        conditions=split_data["conditions"],
        device=device,
        seed=generation_seed,
    )

    rows = []
    for mode, metrics in [("reconstruction", recon_metrics), ("generation", gen_metrics)]:
        row = {"split": split_name, "mode": mode}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_seed_metrics(
    metrics: pd.DataFrame,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize repeated-seed metrics as mean/std tables."""
    if group_columns is None:
        group_columns = ["split", "mode"]

    metric_columns = [
        column
        for column in metrics.columns
        if column not in {*group_columns, "seed"}
        and pd.api.types.is_numeric_dtype(metrics[column])
    ]
    summary = (
        metrics.groupby(group_columns)[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return summary

