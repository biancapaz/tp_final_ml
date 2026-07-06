from __future__ import annotations

import json
import os
import pickle
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "tp_final_ml_matplotlib_cache"),
)

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import pairwise_distances

PROVINCE_ORDER = [
    "buenos_aires", "catamarca", "chaco", "chubut", "cordoba",
    "corrientes", "entre_rios", "formosa", "jujuy", "la_pampa",
    "la_rioja", "mendoza", "misiones", "neuquen", "rio_negro",
    "salta", "san_juan", "san_luis", "santa_cruz", "santa_fe",
    "sgo_del_estero", "tucuman",
]
N_PROVINCES = 22
N_HOURS = 24
PROFILE_DIM = N_PROVINCES * N_HOURS


def get_provinces_root() -> Path:
    """Return the provincial project folder that contains data, src and results."""
    return Path(__file__).resolve().parents[1]


def get_default_splits_dir() -> Path:
    """Return the canonical split directory used by all model scripts."""
    return get_provinces_root() / "data" / "splits"


def get_default_results_dir(model_name: str) -> Path:
    """Return the default output directory for a model run."""
    return get_provinces_root() / "results" / model_name


def set_random_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    """Select a PyTorch device from an explicit name or automatic detection."""
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_array(splits_dir: Path, name: str) -> np.ndarray:
    """Load one NumPy split array as float32."""
    path = splits_dir / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return np.load(path).astype(np.float32)


def load_dates(splits_dir: Path, split_name: str) -> np.ndarray:
    """Load dates for one split."""
    path = splits_dir / f"dates_{split_name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing date file: {path}")
    return np.load(path, allow_pickle=True)


def load_calendar(splits_dir: Path, split_name: str) -> pd.DataFrame:
    """Load calendar features for one temporal split."""
    path = splits_dir / f"calendar_{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing calendar file: {path}")

    calendar = pd.read_csv(path, parse_dates=["fecha"])
    required_columns = {"fecha", "mes", "dia_semana", "es_feriado"}
    missing_columns = required_columns.difference(calendar.columns)
    if missing_columns:
        raise ValueError(f"Calendar {path} is missing columns: {sorted(missing_columns)}")
    return calendar


def load_x_splits(splits_dir: Path) -> dict[str, np.ndarray]:
    """Load X train, validation and test arrays from the split directory."""
    return {
        "train": load_array(splits_dir, "X_train"),
        "val": load_array(splits_dir, "X_val"),
        "test": load_array(splits_dir, "X_test"),
    }


def load_x_scaler(splits_dir: Path) -> Any:
    """Load the X scaler fitted during preprocessing on the train split."""
    path = splits_dir / "x_scaler.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing X scaler file: {path}")
    with open(path, "rb") as file:
        return pickle.load(file)


def print_split_checks(X: dict[str, np.ndarray]) -> None:
    """Print shapes and basic numerical checks before training."""
    for split_name, values in X.items():
        print(
            f"X_{split_name}: shape={values.shape}, "
            f"mean={values.mean():.4f}, std={values.std():.4f}, "
            f"nan_count={np.isnan(values).sum()}"
        )


def validate_x_splits(X: dict[str, np.ndarray]) -> None:
    """Validate X split dimensions and NaN absence before model training."""
    for split_name, values in X.items():
        if values.ndim != 2 or values.shape[1] != PROFILE_DIM:
            raise ValueError(
                f"X_{split_name} must have shape (n_days, {PROFILE_DIM}), got {values.shape}"
            )
        if np.isnan(values).any():
            raise ValueError(f"X_{split_name} contains NaN values")


def inverse_transform_profiles(values: np.ndarray, scaler: Any) -> np.ndarray:
    """Transform scaled profiles back to original MWh units."""
    original_shape = values.shape
    matrix = values.reshape(-1, PROFILE_DIM)
    transformed = scaler.inverse_transform(matrix)
    return transformed.reshape(original_shape).astype(np.float32)


def compute_reconstruction_loss(model_output: np.ndarray, target: np.ndarray) -> float:
    """Compute mean squared reconstruction error in scaled space."""
    return float(np.mean((model_output - target) ** 2))


def coverage_p5_p95(real: np.ndarray, generated: np.ndarray) -> float:
    """Compute empirical coverage of real values inside the generated P5-P95 band."""
    p5 = np.percentile(generated, 5, axis=1)
    p95 = np.percentile(generated, 95, axis=1)
    covered = (real >= p5) & (real <= p95)
    return float(covered.mean())


def sharpness_p5_p95(generated: np.ndarray) -> float:
    """Compute average width of the generated P5-P95 band."""
    p5 = np.percentile(generated, 5, axis=1)
    p95 = np.percentile(generated, 95, axis=1)
    return float(np.mean(p95 - p5))


def mean_profile_rmse(real: np.ndarray, generated: np.ndarray) -> float:
    """Compute RMSE between real profiles and the generated mean profile per day."""
    generated_mean = generated.mean(axis=1)
    return float(np.sqrt(np.mean((generated_mean - real) ** 2)))


def median_gamma(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Estimate an RBF gamma value using the median squared distance heuristic."""
    combined = np.vstack([values_a, values_b])
    distances = pairwise_distances(combined, metric="sqeuclidean")
    upper = distances[np.triu_indices_from(distances, k=1)]
    positive = upper[upper > 0]
    if len(positive) == 0:
        return 1.0
    return float(1.0 / (2.0 * np.median(positive)))


def mmd_rbf(
    real: np.ndarray,
    generated: np.ndarray,
    max_samples: int = 1000,
    seed: int = 42,
) -> float:
    """Compute an RBF-kernel MMD estimate between real and generated profiles."""
    rng = np.random.default_rng(seed)
    real_sample = real
    generated_sample = generated

    if len(real_sample) > max_samples:
        real_indices = rng.choice(len(real_sample), size=max_samples, replace=False)
        real_sample = real_sample[real_indices]
    if len(generated_sample) > max_samples:
        generated_indices = rng.choice(len(generated_sample), size=max_samples, replace=False)
        generated_sample = generated_sample[generated_indices]

    gamma = median_gamma(real_sample, generated_sample)
    xx = np.exp(-gamma * pairwise_distances(real_sample, real_sample, metric="sqeuclidean"))
    yy = np.exp(
        -gamma * pairwise_distances(generated_sample, generated_sample, metric="sqeuclidean")
    )
    xy = np.exp(
        -gamma * pairwise_distances(real_sample, generated_sample, metric="sqeuclidean")
    )
    return float(xx.mean() + yy.mean() - 2.0 * xy.mean())


def province_slice(province: str) -> slice:
    """Return the flattened profile slice corresponding to one province."""
    start = PROVINCE_ORDER.index(province) * N_HOURS
    return slice(start, start + N_HOURS)


def extract_province(x: np.ndarray, province: str) -> np.ndarray:
    """Extract the 24-hour profile for one province from flattened profiles."""
    return x[..., province_slice(province)]


def hour_columns() -> list[str]:
    """Return standardized hour column names for province-hour matrices."""
    return [f"hour_{hour:02d}" for hour in range(1, N_HOURS + 1)]


def fallback_level_counts(fallback_usage: pd.DataFrame) -> dict[str, int]:
    """Count how many test days used each hierarchical fallback level."""
    counts = fallback_usage["fallback_level"].value_counts().to_dict()
    return {
        "n_full": int(counts.get("full", 0)),
        "n_month_weekday": int(counts.get("month_weekday", 0)),
        "n_month": int(counts.get("month", 0)),
        "n_global": int(counts.get("global", 0)),
    }


def evaluate_by_province_from_original(
    real: np.ndarray,
    generated: np.ndarray,
    seed: int,
    mmd_max_samples: int,
) -> pd.DataFrame:
    """Evaluate generated scenarios by province in original MWh scale."""
    province_rows = []

    for province in PROVINCE_ORDER:
        real_province = extract_province(real, province)
        generated_province = extract_province(generated, province)
        generated_province_flat = generated_province.reshape(-1, N_HOURS)
        rmse = mean_profile_rmse(real_province, generated_province)
        mean_demand = float(real_province.mean())
        coverage = coverage_p5_p95(real_province, generated_province)
        sharpness = sharpness_p5_p95(generated_province)
        province_rows.append(
            {
                "province": province,
                "rmse": rmse,
                "mean_profile_rmse_mwh": rmse,
                "mean_demand": mean_demand,
                "mean_demand_mwh": mean_demand,
                "nrmse": float(rmse / max(mean_demand, 1e-8)),
                "coverage": coverage,
                "coverage_p5_p95": coverage,
                "sharpness": sharpness,
                "sharpness_p5_p95_mwh": sharpness,
                "mmd_rbf": mmd_rbf(
                    real_province,
                    generated_province_flat,
                    max_samples=mmd_max_samples,
                    seed=seed,
                ),
            }
        )

    return pd.DataFrame(province_rows)


def evaluate_overall_generated_scenarios(
    real_scaled: np.ndarray,
    generated_scaled: np.ndarray,
    scaler: Any,
    seed: int,
    mmd_max_samples: int,
) -> pd.DataFrame:
    """Evaluate generated scenarios overall in original MWh scale."""
    real = inverse_transform_profiles(real_scaled, scaler)
    generated = inverse_transform_profiles(generated_scaled, scaler)
    generated_flat = generated.reshape(-1, PROFILE_DIM)
    overall = {
        "coverage_p5_p95": coverage_p5_p95(real, generated),
        "sharpness_p5_p95_mwh": sharpness_p5_p95(generated),
        "mean_profile_rmse_mwh": mean_profile_rmse(real, generated),
        "mmd_rbf": mmd_rbf(real, generated_flat, max_samples=mmd_max_samples, seed=seed),
        "n_days": int(real.shape[0]),
        "n_scenarios_per_day": int(generated.shape[1]),
    }
    return pd.DataFrame([overall])


def evaluate_generated_scenarios(
    real_scaled: np.ndarray,
    generated_scaled: np.ndarray,
    scaler: Any,
    seed: int,
    mmd_max_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate generated scenarios overall and by province in original MWh scale."""
    overall_metrics = evaluate_overall_generated_scenarios(
        real_scaled=real_scaled,
        generated_scaled=generated_scaled,
        scaler=scaler,
        seed=seed,
        mmd_max_samples=mmd_max_samples,
    )
    real = inverse_transform_profiles(real_scaled, scaler)
    generated = inverse_transform_profiles(generated_scaled, scaler)
    province_metrics = evaluate_by_province_from_original(
        real=real,
        generated=generated,
        seed=seed,
        mmd_max_samples=mmd_max_samples,
    )
    return overall_metrics, province_metrics


def evaluate_by_month(
    real_scaled: np.ndarray,
    generated_scaled: np.ndarray,
    scaler: Any,
    calendar_test: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate generated scenarios by calendar month in original MWh scale."""
    real = inverse_transform_profiles(real_scaled, scaler)
    generated = inverse_transform_profiles(generated_scaled, scaler)
    rows = []

    for month in sorted(calendar_test["mes"].unique()):
        mask = calendar_test["mes"].to_numpy() == month
        real_month = real[mask]
        generated_month = generated[mask]
        rows.append(
            {
                "mes": int(month),
                "coverage": coverage_p5_p95(real_month, generated_month),
                "sharpness": sharpness_p5_p95(generated_month),
                "rmse_mean_profile": mean_profile_rmse(real_month, generated_month),
                "n_days": int(mask.sum()),
            }
        )

    return pd.DataFrame(rows)


def compute_province_hour_matrices(
    real_scaled: np.ndarray,
    generated_scaled: np.ndarray,
    scaler: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute coverage and RMSE matrices with shape provinces by hours."""
    real = inverse_transform_profiles(real_scaled, scaler)
    generated = inverse_transform_profiles(generated_scaled, scaler)
    lower = np.percentile(generated, 5, axis=1)
    upper = np.percentile(generated, 95, axis=1)
    generated_mean = generated.mean(axis=1)

    coverage_flat = ((real >= lower) & (real <= upper)).mean(axis=0)
    rmse_flat = np.sqrt(np.mean((generated_mean - real) ** 2, axis=0))
    coverage_matrix = coverage_flat.reshape(N_PROVINCES, N_HOURS)
    rmse_matrix = rmse_flat.reshape(N_PROVINCES, N_HOURS)

    return (
        pd.DataFrame(coverage_matrix, index=PROVINCE_ORDER, columns=hour_columns()),
        pd.DataFrame(rmse_matrix, index=PROVINCE_ORDER, columns=hour_columns()),
    )


def save_extended_metrics(
    real_scaled: np.ndarray,
    generated_scaled: np.ndarray,
    scaler: Any,
    calendar_test: pd.DataFrame,
    output_dir: Path,
    seed: int,
    mmd_max_samples: int,
) -> dict[str, Path]:
    """Save monthly, province-hour and province NRMSE metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_month = evaluate_by_month(real_scaled, generated_scaled, scaler, calendar_test)
    _, metrics_by_province = evaluate_generated_scenarios(
        real_scaled=real_scaled,
        generated_scaled=generated_scaled,
        scaler=scaler,
        seed=seed,
        mmd_max_samples=mmd_max_samples,
    )
    coverage_by_hour, rmse_by_hour = compute_province_hour_matrices(
        real_scaled=real_scaled,
        generated_scaled=generated_scaled,
        scaler=scaler,
    )

    metrics_by_month.to_csv(output_dir / "metrics_by_month.csv", index=False)
    metrics_by_province.to_csv(output_dir / "metrics_by_province.csv", index=False)
    coverage_by_hour.to_csv(output_dir / "coverage_by_province_hour.csv")
    rmse_by_hour.to_csv(output_dir / "rmse_by_province_hour.csv")

    return {
        "metrics_by_month": output_dir / "metrics_by_month.csv",
        "metrics_by_province": output_dir / "metrics_by_province.csv",
        "coverage_by_province_hour": output_dir / "coverage_by_province_hour.csv",
        "rmse_by_province_hour": output_dir / "rmse_by_province_hour.csv",
    }


def system_hourly_profile(profile: np.ndarray) -> np.ndarray:
    """Aggregate one flattened all-province profile into a 24-hour system profile."""
    return profile.reshape(N_PROVINCES, N_HOURS).sum(axis=0)


def save_generation_plots(
    real_scaled: np.ndarray,
    generated_scaled: np.ndarray,
    scaler: Any,
    dates: np.ndarray,
    output_dir: Path,
    n_days: int = 4,
) -> None:
    """Save real vs generated system-level plots for representative test days."""
    output_dir.mkdir(parents=True, exist_ok=True)
    real = inverse_transform_profiles(real_scaled, scaler)
    generated = inverse_transform_profiles(generated_scaled, scaler)
    selected = np.linspace(0, len(real) - 1, num=min(n_days, len(real)), dtype=int)
    hours = np.arange(1, N_HOURS + 1)

    for index in selected:
        real_system = system_hourly_profile(real[index])
        generated_system = np.array(
            [system_hourly_profile(profile) for profile in generated[index]]
        )
        lower = np.percentile(generated_system, 5, axis=0)
        upper = np.percentile(generated_system, 95, axis=0)
        mean = generated_system.mean(axis=0)
        date_label = pd.to_datetime(dates[index]).strftime("%Y-%m-%d")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(hours, real_system, label="Real", color="#2B6CB0", linewidth=2.0)
        ax.plot(hours, mean, label="Generated mean", color="#C53030", linewidth=2.0)
        ax.fill_between(hours, lower, upper, color="#C53030", alpha=0.18, label="P5-P95")
        ax.set_title(f"Real vs generated system profile - {date_label}")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Demand (MWh)")
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.grid(linewidth=0.4, alpha=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"real_vs_generated_{date_label}.png", dpi=150)
        plt.close(fig)


def load_result_table(output_dir: Path, filename: str) -> pd.DataFrame:
    """Load one CSV result table from a model output directory."""
    path = output_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing result table: {path}")
    return pd.read_csv(path)


def list_generation_plot_paths(output_dir: Path, max_plots: int = 4) -> list[Path]:
    """Return saved generation plot paths for notebook display."""
    plot_dir = output_dir / "plots"
    if not plot_dir.exists():
        return []
    return sorted(plot_dir.glob("real_vs_generated_*.png"))[:max_plots]


def show_or_close_figure(fig: Any) -> None:
    """Show a figure in interactive backends and close it in terminal runs."""
    if matplotlib.get_backend().lower() == "agg":
        plt.close(fig)
        return
    plt.show()


def plot_training_history(
    history: pd.DataFrame,
    train_column: str,
    val_column: str,
    title: str,
    ylabel: str,
) -> None:
    """Plot train and validation curves from a saved training history table."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["epoch"], history[train_column], label="Train", color="#2B6CB0")
    ax.plot(history["epoch"], history[val_column], label="Validation", color="#C53030")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(linewidth=0.4, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    show_or_close_figure(fig)


def plot_bandwidth_validation(bandwidth_results: pd.DataFrame) -> None:
    """Plot validation log-likelihood by KDE bandwidth."""
    ordered = bandwidth_results.sort_values("bandwidth")
    score_column = (
        "val_log_likelihood"
        if "val_log_likelihood" in ordered.columns
        else "val_mean_log_likelihood"
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        ordered["bandwidth"],
        ordered[score_column],
        marker="o",
        color="#2B6CB0",
    )
    ax.set_title("KDE bandwidth selection on validation")
    ax.set_xlabel("Bandwidth")
    ax.set_ylabel("Mean validation log-likelihood")
    ax.grid(linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    show_or_close_figure(fig)


def plot_metric_by_province(
    province_metrics: pd.DataFrame,
    metric: str,
    title: str,
) -> None:
    """Plot one metric by province from a saved evaluation table."""
    ordered = province_metrics.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(ordered["province"], ordered[metric], color="#2B6CB0")
    ax.set_title(title)
    ax.set_xlabel(metric)
    ax.set_ylabel("Province")
    ax.grid(axis="x", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    show_or_close_figure(fig)


def plot_month_metrics(month_metrics: pd.DataFrame, title_prefix: str) -> None:
    """Plot coverage, RMSE and sharpness by month."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    plots = [
        ("coverage", "Coverage P5-P95", "#2B6CB0"),
        ("rmse_mean_profile", "RMSE mean profile (MWh)", "#C53030"),
        ("sharpness", "Sharpness P5-P95 (MWh)", "#6A994E"),
    ]

    for ax, (column, ylabel, color) in zip(axes, plots):
        ax.plot(month_metrics["mes"], month_metrics[column], marker="o", color=color)
        ax.set_title(ylabel)
        ax.set_xlabel("Month")
        ax.set_ylabel(ylabel)
        ax.set_xticks(month_metrics["mes"])
        ax.grid(linewidth=0.4, alpha=0.5)

    fig.suptitle(title_prefix)
    fig.tight_layout()
    show_or_close_figure(fig)


def plot_province_hour_heatmap(
    matrix: pd.DataFrame,
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
) -> None:
    """Plot a province-hour matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(11, 7))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Province")
    ax.set_xticks(np.arange(N_HOURS))
    ax.set_xticklabels(range(1, N_HOURS + 1))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    show_or_close_figure(fig)


def plot_nrmse_by_province(province_metrics: pd.DataFrame, title: str) -> None:
    """Plot province-level NRMSE as a horizontal bar chart."""
    ordered = province_metrics.sort_values("nrmse", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(ordered["province"], ordered["nrmse"], color="#6A994E")
    ax.set_title(title)
    ax.set_xlabel("NRMSE")
    ax.set_ylabel("Province")
    ax.grid(axis="x", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    show_or_close_figure(fig)


def plot_latent_boxplot(values: np.ndarray, title: str, ylabel: str) -> None:
    """Plot a boxplot for latent values by dimension."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot([values[:, index] for index in range(values.shape[1])])
    ax.set_title(title)
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    show_or_close_figure(fig)


def plot_latent_kl(latent_kl: pd.DataFrame) -> None:
    """Plot KL divergence by latent dimension."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(latent_kl["latent_dim"], latent_kl["kl"], color="#2B6CB0")
    ax.set_title("KL divergence by latent dimension")
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("KL")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    show_or_close_figure(fig)


def plot_real_reconstructed_prior_profiles(
    real_scaled: np.ndarray,
    reconstructed_scaled: np.ndarray,
    prior_scaled: np.ndarray,
    scaler: Any,
    dates: np.ndarray,
) -> None:
    """Plot real, reconstructed and prior-generated system profiles."""
    real = inverse_transform_profiles(real_scaled, scaler)
    reconstructed = inverse_transform_profiles(reconstructed_scaled, scaler)
    prior = inverse_transform_profiles(prior_scaled, scaler)
    n_examples = len(real)
    hours = np.arange(1, N_HOURS + 1)
    fig, axes = plt.subplots(1, n_examples, figsize=(5 * n_examples, 4), sharey=True)
    if n_examples == 1:
        axes = [axes]

    for index, ax in enumerate(axes):
        real_system = system_hourly_profile(real[index])
        reconstructed_system = system_hourly_profile(reconstructed[index])
        prior_system = system_hourly_profile(prior[index])
        date_label = pd.to_datetime(dates[index]).strftime("%Y-%m-%d")
        ax.plot(hours, real_system, label="Real", color="#2B6CB0", linewidth=2)
        ax.plot(hours, reconstructed_system, label="Reconstructed", color="#6A994E", linewidth=2)
        ax.plot(hours, prior_system, label="Prior sample", color="#C53030", linewidth=2)
        ax.set_title(date_label)
        ax.set_xlabel("Hour")
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.grid(linewidth=0.4, alpha=0.5)
        if index == 0:
            ax.set_ylabel("Demand (MWh)")
            ax.legend()

    fig.suptitle("Real profiles, reconstructions and prior samples")
    fig.tight_layout()
    show_or_close_figure(fig)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save a JSON file with stable formatting."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def save_interpretation(text: str, path: Path) -> None:
    """Save a short interpretation text next to model outputs."""
    with open(path, "w", encoding="utf-8") as file:
        file.write(text.strip() + "\n")
