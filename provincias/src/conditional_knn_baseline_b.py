from __future__ import annotations

import argparse
import json
import os
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
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from preprocessing_b import PROVINCE_ORDER, run_preprocessing_b


N_HOURS = 24


def set_random_seed(seed: int) -> None:
    """Set the NumPy random seed used by this baseline."""
    np.random.seed(seed)


def get_provinces_root() -> Path:
    """Return the provincial project folder that contains data, src and results."""
    return Path(__file__).resolve().parents[1]


def get_default_processed_dir() -> Path:
    """Return the processed data directory used by Architecture B."""
    return get_provinces_root() / "data" / "processed"


def get_default_splits_dir() -> Path:
    """Return the split directory where Architecture B arrays are cached."""
    return get_provinces_root() / "data" / "splits"


def get_default_output_dir() -> Path:
    """Return the default output directory for the conditional kNN baseline."""
    return get_provinces_root() / "results" / "conditional_knn_b"


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save a JSON file with stable formatting."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def load_cvae_b_metrics(cvae_metrics_path: Path) -> dict[str, Any] | None:
    """Load saved CVAE-B metrics if the file exists."""
    if not cvae_metrics_path.exists():
        return None
    with open(cvae_metrics_path, "r", encoding="utf-8") as file:
        return json.load(file)


def fit_knn_index(C_train: np.ndarray, max_k: int) -> NearestNeighbors:
    """Fit a nearest-neighbor index over training contexts only."""
    model = NearestNeighbors(n_neighbors=max_k, metric="euclidean")
    model.fit(C_train)
    return model


def compute_neighbors(
    knn_model: NearestNeighbors,
    C_target: np.ndarray,
    max_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return distances and training indices for the nearest contexts."""
    distances, indices = knn_model.kneighbors(C_target, n_neighbors=max_k, return_distance=True)
    return distances.astype(np.float32), indices.astype(np.int64)


def softmax_distance_weights(distances: np.ndarray, tau_scale: float) -> np.ndarray:
    """Convert neighbor distances into softmax weights with a local temperature."""
    local_tau = tau_scale * np.median(distances, axis=1, keepdims=True)
    positive = distances[distances > 0]
    fallback_tau = float(np.median(positive)) if len(positive) else 1.0
    local_tau = np.where(local_tau > 1e-8, local_tau, fallback_tau)
    local_tau = np.maximum(local_tau, 1e-8)
    logits = -(distances**2) / (2.0 * local_tau**2)
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights_sum = weights.sum(axis=1, keepdims=True)
    return weights / np.maximum(weights_sum, 1e-12)


def sample_weighted_positions(weights: np.ndarray, n_scenarios: int, rng: np.random.Generator) -> np.ndarray:
    """Sample neighbor positions independently for each target context."""
    sampled = np.empty((weights.shape[0], n_scenarios), dtype=np.int64)
    cumulative = np.cumsum(weights, axis=1)

    for row_index in range(weights.shape[0]):
        draws = rng.random(n_scenarios)
        sampled[row_index] = np.searchsorted(cumulative[row_index], draws, side="right")

    return np.minimum(sampled, weights.shape[1] - 1)


def generate_knn_scenarios(
    X_train: np.ndarray,
    neighbor_distances: np.ndarray,
    neighbor_indices: np.ndarray,
    k: int,
    tau_scale: float,
    bandwidth_x: float,
    n_scenarios: int,
    seed: int,
) -> np.ndarray:
    """Generate scaled demand scenarios by sampling training profiles from local neighbors."""
    rng = np.random.default_rng(seed)
    distances_k = neighbor_distances[:, :k]
    indices_k = neighbor_indices[:, :k]
    weights = softmax_distance_weights(distances_k, tau_scale)
    sampled_positions = sample_weighted_positions(weights, n_scenarios, rng)
    sampled_train_indices = np.take_along_axis(indices_k, sampled_positions, axis=1)
    scenarios = X_train[sampled_train_indices].astype(np.float32)

    if bandwidth_x > 0:
        noise = rng.normal(0.0, bandwidth_x, size=scenarios.shape).astype(np.float32)
        scenarios = scenarios + noise

    return scenarios.astype(np.float32)


def inverse_transform_scenarios(
    generated_scaled: np.ndarray,
    x_scaler: Any,
    chunk_samples: int = 1000,
) -> np.ndarray:
    """Inverse-transform generated scenarios to MWh without a large temporary matrix."""
    generated_mwh = np.empty(generated_scaled.shape, dtype=np.float32)

    for start in range(0, generated_scaled.shape[0], chunk_samples):
        end = min(start + chunk_samples, generated_scaled.shape[0])
        chunk = generated_scaled[start:end].reshape(-1, generated_scaled.shape[-1])
        transformed = x_scaler.inverse_transform(chunk)
        generated_mwh[start:end] = transformed.reshape(generated_scaled[start:end].shape).astype(np.float32)

    return generated_mwh


def mmd_rbf_cv_style(
    real_mwh: np.ndarray,
    generated_mwh: np.ndarray,
    seed: int,
    sigma: float = 1.0,
    max_samples: int = 500,
) -> float:
    """Compute the same RBF MMD style used in the CVAE evaluation notebook."""
    rng = np.random.default_rng(seed)
    n_samples = min(max_samples, len(real_mwh))
    selected = rng.choice(len(real_mwh), n_samples, replace=False)
    scenario_ids = rng.integers(0, generated_mwh.shape[1], size=n_samples)
    real_sample = real_mwh[selected]
    generated_sample = generated_mwh[selected, scenario_ids, :]

    def rbf(values_a: np.ndarray, values_b: np.ndarray) -> np.ndarray:
        squared_distances = pairwise_distances(values_a, values_b, metric="sqeuclidean")
        return np.exp(-squared_distances / (2.0 * sigma**2))

    return float(rbf(real_sample, real_sample).mean() + rbf(generated_sample, generated_sample).mean() - 2.0 * rbf(real_sample, generated_sample).mean())


def compute_metrics_mwh(
    real_mwh: np.ndarray,
    generated_mwh: np.ndarray,
    seed: int,
) -> dict[str, float]:
    """Compute point and generative metrics in original MWh units."""
    median_profile = np.median(generated_mwh, axis=1)
    mean_profile = generated_mwh.mean(axis=1)
    lower = np.percentile(generated_mwh, 5, axis=1)
    upper = np.percentile(generated_mwh, 95, axis=1)
    energy_real = real_mwh.sum(axis=1)
    energy_pred = median_profile.sum(axis=1)
    real_std = real_mwh.std(axis=0).mean()
    generated_std = generated_mwh.std(axis=1).mean()

    return {
        "mae_global": float(np.abs(real_mwh - median_profile).mean()),
        "rmse_global": float(np.sqrt(np.mean((real_mwh - median_profile) ** 2))),
        "mean_profile_rmse": float(np.sqrt(np.mean((real_mwh - mean_profile) ** 2))),
        "mape_energy": float(np.mean(np.abs(energy_real - energy_pred) / np.maximum(energy_real, 1e-8)) * 100.0),
        "coverage": float(((real_mwh >= lower) & (real_mwh <= upper)).mean() * 100.0),
        "sharpness": float((upper - lower).mean()),
        "diversity_ratio": float(generated_std / (real_std + 1e-8)),
        "mmd": mmd_rbf_cv_style(real_mwh, generated_mwh, seed=seed),
    }


def evaluate_scenarios_scaled(
    X_real: np.ndarray,
    generated_scaled: np.ndarray,
    x_scaler: Any,
    seed: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Inverse-transform scaled arrays and compute all metrics."""
    real_mwh = x_scaler.inverse_transform(X_real).astype(np.float32)
    generated_mwh = inverse_transform_scenarios(generated_scaled, x_scaler)
    metrics = compute_metrics_mwh(real_mwh, generated_mwh, seed=seed)
    return metrics, real_mwh, generated_mwh


def metrics_by_province_b(
    real_mwh: np.ndarray,
    generated_mwh: np.ndarray,
    provinces: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    """Compute metrics separately for each province in Architecture B format."""
    rows = []

    for province in PROVINCE_ORDER:
        mask = provinces == province
        if not mask.any():
            continue
        metrics = compute_metrics_mwh(real_mwh[mask], generated_mwh[mask], seed=seed)
        metrics["province"] = province
        metrics["n_samples"] = int(mask.sum())
        rows.append(metrics)

    return pd.DataFrame(rows).sort_values("rmse_global", ascending=False).reset_index(drop=True)


def metrics_by_month_b(
    real_mwh: np.ndarray,
    generated_mwh: np.ndarray,
    dates: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    """Compute metrics separately for each calendar month."""
    months = pd.to_datetime(dates).month.to_numpy()
    rows = []

    for month in sorted(pd.unique(months)):
        mask = months == month
        metrics = compute_metrics_mwh(real_mwh[mask], generated_mwh[mask], seed=seed)
        metrics["month"] = int(month)
        metrics["n_samples"] = int(mask.sum())
        rows.append(metrics)

    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def province_hour_metrics_b(
    real_mwh: np.ndarray,
    generated_mwh: np.ndarray,
    provinces: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute coverage and RMSE matrices by province and hour."""
    lower = np.percentile(generated_mwh, 5, axis=1)
    upper = np.percentile(generated_mwh, 95, axis=1)
    mean_profile = generated_mwh.mean(axis=1)
    hour_columns = [f"hour_{hour:02d}" for hour in range(1, N_HOURS + 1)]
    coverage_rows = []
    rmse_rows = []

    for province in PROVINCE_ORDER:
        mask = provinces == province
        coverage = ((real_mwh[mask] >= lower[mask]) & (real_mwh[mask] <= upper[mask])).mean(axis=0) * 100.0
        rmse = np.sqrt(np.mean((mean_profile[mask] - real_mwh[mask]) ** 2, axis=0))
        coverage_rows.append(coverage)
        rmse_rows.append(rmse)

    coverage_matrix = pd.DataFrame(coverage_rows, index=PROVINCE_ORDER, columns=hour_columns)
    rmse_matrix = pd.DataFrame(rmse_rows, index=PROVINCE_ORDER, columns=hour_columns)
    return coverage_matrix, rmse_matrix


def run_validation_grid(
    X_train: np.ndarray,
    X_val: np.ndarray,
    neighbor_distances_val: np.ndarray,
    neighbor_indices_val: np.ndarray,
    x_scaler: Any,
    k_grid: list[int],
    tau_scale_grid: list[float],
    bandwidth_x_grid: list[float],
    n_scenarios: int,
    seed: int,
) -> pd.DataFrame:
    """Evaluate the conditional kNN hyperparameter grid on validation only."""
    rows = []
    run_index = 0

    for k in k_grid:
        for tau_scale in tau_scale_grid:
            for bandwidth_x in bandwidth_x_grid:
                scenario_seed = seed + run_index
                generated_val = generate_knn_scenarios(
                    X_train=X_train,
                    neighbor_distances=neighbor_distances_val,
                    neighbor_indices=neighbor_indices_val,
                    k=k,
                    tau_scale=tau_scale,
                    bandwidth_x=bandwidth_x,
                    n_scenarios=n_scenarios,
                    seed=scenario_seed,
                )
                metrics, _, _ = evaluate_scenarios_scaled(
                    X_real=X_val,
                    generated_scaled=generated_val,
                    x_scaler=x_scaler,
                    seed=scenario_seed,
                )
                row = {
                    "k": int(k),
                    "tau_scale": float(tau_scale),
                    "bandwidth_x": float(bandwidth_x),
                    "n_scenarios_validation": int(n_scenarios),
                    "coverage_gap": abs(metrics["coverage"] - 90.0),
                }
                row.update({f"val_{key}": value for key, value in metrics.items()})
                rows.append(row)
                print(
                    f"k={k:3d} | tau_scale={tau_scale:.2f} | bandwidth_x={bandwidth_x:.2f} | "
                    f"coverage={metrics['coverage']:.2f} | sharpness={metrics['sharpness']:.2f} | "
                    f"rmse={metrics['rmse_global']:.2f}"
                )
                run_index += 1

    return pd.DataFrame(rows)


def select_validation_configuration(validation_grid: pd.DataFrame) -> pd.Series:
    """Select the final configuration using validation metrics only."""
    ordered = validation_grid.sort_values(
        ["coverage_gap", "val_sharpness", "val_mean_profile_rmse", "val_mmd"],
        ascending=[True, True, True, True],
    )
    return ordered.iloc[0]


def save_generation_plots_b(
    real_mwh: np.ndarray,
    generated_mwh: np.ndarray,
    dates: np.ndarray,
    provinces: np.ndarray,
    output_dir: Path,
    n_days: int,
) -> None:
    """Save real-vs-generated plots for selected dates and provinces."""
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_dates = pd.to_datetime(pd.Series(dates)).drop_duplicates().sort_values().to_numpy()
    selected_positions = np.linspace(0, len(unique_dates) - 1, num=min(n_days, len(unique_dates)), dtype=int)
    selected_dates = unique_dates[selected_positions]
    provinces_to_show = ["buenos_aires", "cordoba", "santa_fe", "mendoza", "neuquen", "santa_cruz"]
    hours = np.arange(1, N_HOURS + 1)
    dates_ts = pd.to_datetime(dates)

    for selected_date in selected_dates:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
        date_label = pd.to_datetime(selected_date).strftime("%Y-%m-%d")

        for axis, province in zip(axes.flat, provinces_to_show):
            mask = (dates_ts == selected_date) & (provinces == province)
            if not mask.any():
                axis.axis("off")
                continue
            index = int(np.where(mask)[0][0])
            lower = np.percentile(generated_mwh[index], 5, axis=0)
            upper = np.percentile(generated_mwh[index], 95, axis=0)
            median = np.median(generated_mwh[index], axis=0)
            axis.fill_between(hours, lower, upper, color="#2B6CB0", alpha=0.18, label="P5-P95")
            axis.plot(hours, median, color="#2B6CB0", linewidth=2.0, label="Median generated")
            axis.plot(hours, real_mwh[index], color="#C53030", linewidth=2.0, linestyle="--", label="Real")
            axis.set_title(province.replace("_", " ").title())
            axis.set_xlabel("Hour")
            axis.set_ylabel("Demand (MWh)")
            axis.set_xticks([1, 6, 12, 18, 24])
            axis.grid(linewidth=0.4, alpha=0.5)
            axis.legend(fontsize=7)

        fig.suptitle(f"Conditional kNN-B - real vs generated profiles - {date_label}")
        fig.tight_layout()
        fig.savefig(output_dir / f"real_vs_generated_{date_label}.png", dpi=150)
        plt.close(fig)


def build_cvae_comparison(
    knn_metrics: dict[str, float],
    cvae_metrics_path: Path,
) -> pd.DataFrame:
    """Build a compact comparison table between CVAE-B and the kNN baseline."""
    rows = []
    cvae_metrics = load_cvae_b_metrics(cvae_metrics_path)

    if cvae_metrics is not None and "test" in cvae_metrics:
        cvae_test = cvae_metrics["test"]
        rows.append(
            {
                "model": "CVAE-B",
                "mae_global": cvae_test.get("mae_global"),
                "rmse_global": cvae_test.get("rmse_global"),
                "mape_energy": cvae_test.get("mape_energy"),
                "coverage": cvae_test.get("coverage"),
                "sharpness": cvae_test.get("sharpness"),
                "diversity_ratio": cvae_test.get("diversity_ratio"),
                "mmd": cvae_test.get("mmd"),
            }
        )

    rows.append(
        {
            "model": "Conditional kNN-B",
            "mae_global": knn_metrics["mae_global"],
            "rmse_global": knn_metrics["rmse_global"],
            "mape_energy": knn_metrics["mape_energy"],
            "coverage": knn_metrics["coverage"],
            "sharpness": knn_metrics["sharpness"],
            "diversity_ratio": knn_metrics["diversity_ratio"],
            "mmd": knn_metrics["mmd"],
        }
    )
    return pd.DataFrame(rows)


def save_interpretation(path: Path) -> None:
    """Save the report interpretation requested for the kNN baseline."""
    text = """
Este baseline permite evaluar cuanto de la distribucion condicional puede explicarse unicamente con C. Si el CVAE-B mejora frente a este baseline, entonces la complejidad adicional del espacio latente se justifica porque captura variabilidad residual no explicada por el contexto. Si el kNN condicional obtiene metricas iguales o mejores, eso indicaria que el contexto disponible ya contiene gran parte de la informacion necesaria y que el CVAE-B no aporta una mejora generativa clara.
"""
    path.write_text(text.strip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the conditional kNN baseline."""
    parser = argparse.ArgumentParser(description="Run the Architecture B conditional kNN baseline.")
    parser.add_argument("--processed-dir", type=Path, default=get_default_processed_dir())
    parser.add_argument("--splits-dir", type=Path, default=get_default_splits_dir())
    parser.add_argument("--output-dir", type=Path, default=get_default_output_dir())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-grid", type=int, nargs="+", default=[10, 25, 50, 100])
    parser.add_argument("--tau-scale-grid", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--bandwidth-x-grid", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.10])
    parser.add_argument("--n-scenarios-validation", type=int, default=200)
    parser.add_argument("--n-scenarios-test", type=int, default=500)
    parser.add_argument("--plot-days", type=int, default=4)
    return parser.parse_args()


def run_conditional_knn_baseline_b(args: argparse.Namespace) -> dict[str, Path]:
    """Run preprocessing, validation selection and final test evaluation for kNN-B."""
    set_random_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_b = run_preprocessing_b(processed_dir=args.processed_dir, save_dir=args.splits_dir)
    X_train = data_b["X_train"].astype(np.float32)
    X_val = data_b["X_val"].astype(np.float32)
    X_test = data_b["X_test"].astype(np.float32)
    C_train = data_b["C_train"].astype(np.float32)
    C_val = data_b["C_val"].astype(np.float32)
    C_test = data_b["C_test"].astype(np.float32)
    x_scaler = data_b["x_scaler"]
    max_k = max(args.k_grid)

    knn_model = fit_knn_index(C_train, max_k=max_k)
    neighbor_distances_val, neighbor_indices_val = compute_neighbors(knn_model, C_val, max_k=max_k)
    validation_grid = run_validation_grid(
        X_train=X_train,
        X_val=X_val,
        neighbor_distances_val=neighbor_distances_val,
        neighbor_indices_val=neighbor_indices_val,
        x_scaler=x_scaler,
        k_grid=list(args.k_grid),
        tau_scale_grid=list(args.tau_scale_grid),
        bandwidth_x_grid=list(args.bandwidth_x_grid),
        n_scenarios=int(args.n_scenarios_validation),
        seed=int(args.seed),
    )
    validation_grid.to_csv(args.output_dir / "validation_grid.csv", index=False)
    selected = select_validation_configuration(validation_grid)
    neighbor_distances_test, neighbor_indices_test = compute_neighbors(knn_model, C_test, max_k=max_k)
    generated_test = generate_knn_scenarios(
        X_train=X_train,
        neighbor_distances=neighbor_distances_test,
        neighbor_indices=neighbor_indices_test,
        k=int(selected["k"]),
        tau_scale=float(selected["tau_scale"]),
        bandwidth_x=float(selected["bandwidth_x"]),
        n_scenarios=int(args.n_scenarios_test),
        seed=int(args.seed) + 10_000,
    )
    test_metrics, real_test_mwh, generated_test_mwh = evaluate_scenarios_scaled(
        X_real=X_test,
        generated_scaled=generated_test,
        x_scaler=x_scaler,
        seed=int(args.seed) + 10_000,
    )
    metrics_overall = pd.DataFrame(
        [
            {
                "model": "conditional_knn_b",
                "k": int(selected["k"]),
                "tau_scale": float(selected["tau_scale"]),
                "bandwidth_x": float(selected["bandwidth_x"]),
                "n_scenarios_test": int(args.n_scenarios_test),
                **test_metrics,
            }
        ]
    )
    selection_comparison = pd.DataFrame(
        [
            {
                "selection_criterion": "validation_generative",
                "k": int(selected["k"]),
                "tau_scale": float(selected["tau_scale"]),
                "bandwidth_x": float(selected["bandwidth_x"]),
                "coverage_gap": float(selected["coverage_gap"]),
                "val_mae_global": float(selected["val_mae_global"]),
                "val_rmse_global": float(selected["val_rmse_global"]),
                "val_mean_profile_rmse": float(selected["val_mean_profile_rmse"]),
                "val_mape_energy": float(selected["val_mape_energy"]),
                "val_coverage": float(selected["val_coverage"]),
                "val_sharpness": float(selected["val_sharpness"]),
                "val_diversity_ratio": float(selected["val_diversity_ratio"]),
                "val_mmd": float(selected["val_mmd"]),
                "test_mae_global": test_metrics["mae_global"],
                "test_rmse_global": test_metrics["rmse_global"],
                "test_mean_profile_rmse": test_metrics["mean_profile_rmse"],
                "test_mape_energy": test_metrics["mape_energy"],
                "test_coverage": test_metrics["coverage"],
                "test_sharpness": test_metrics["sharpness"],
                "test_diversity_ratio": test_metrics["diversity_ratio"],
                "test_mmd": test_metrics["mmd"],
            }
        ]
    )
    province_metrics = metrics_by_province_b(
        real_mwh=real_test_mwh,
        generated_mwh=generated_test_mwh,
        provinces=data_b["provinces_test"],
        seed=int(args.seed) + 10_000,
    )
    month_metrics = metrics_by_month_b(
        real_mwh=real_test_mwh,
        generated_mwh=generated_test_mwh,
        dates=data_b["dates_test"],
        seed=int(args.seed) + 10_000,
    )
    coverage_by_hour, rmse_by_hour = province_hour_metrics_b(
        real_mwh=real_test_mwh,
        generated_mwh=generated_test_mwh,
        provinces=data_b["provinces_test"],
    )
    cvae_comparison = build_cvae_comparison(
        knn_metrics=test_metrics,
        cvae_metrics_path=get_provinces_root() / "results" / "cvae_b" / "cvae_b_metrics.json",
    )

    metrics_overall.to_csv(args.output_dir / "metrics_overall.csv", index=False)
    selection_comparison.to_csv(args.output_dir / "selection_comparison.csv", index=False)
    province_metrics.to_csv(args.output_dir / "metrics_by_province.csv", index=False)
    month_metrics.to_csv(args.output_dir / "metrics_by_month.csv", index=False)
    coverage_by_hour.to_csv(args.output_dir / "coverage_by_province_hour.csv")
    rmse_by_hour.to_csv(args.output_dir / "rmse_by_province_hour.csv")
    cvae_comparison.to_csv(args.output_dir / "cvae_b_comparison.csv", index=False)
    save_generation_plots_b(
        real_mwh=real_test_mwh,
        generated_mwh=generated_test_mwh,
        dates=data_b["dates_test"],
        provinces=data_b["provinces_test"],
        output_dir=args.output_dir / "plots",
        n_days=int(args.plot_days),
    )
    save_json(
        {
            "model": "conditional_knn_baseline_b",
            "processed_dir": str(args.processed_dir),
            "splits_dir": str(args.splits_dir),
            "k_grid": list(args.k_grid),
            "tau_scale_grid": list(args.tau_scale_grid),
            "bandwidth_x_grid": list(args.bandwidth_x_grid),
            "selected_k": int(selected["k"]),
            "selected_tau_scale": float(selected["tau_scale"]),
            "selected_bandwidth_x": float(selected["bandwidth_x"]),
            "selection_criterion": "coverage closest to 90, then lower sharpness, lower mean-profile RMSE and lower MMD on validation",
            "n_scenarios_validation": int(args.n_scenarios_validation),
            "n_scenarios_test": int(args.n_scenarios_test),
            "seed": int(args.seed),
            "uses_province_label_as_input": False,
        },
        args.output_dir / "config.json",
    )
    save_interpretation(args.output_dir / "interpretation.txt")
    print(f"Selected k={int(selected['k'])}, tau_scale={float(selected['tau_scale']):.2f}, bandwidth_x={float(selected['bandwidth_x']):.2f}")
    print(f"Saved conditional kNN-B outputs to: {args.output_dir}")
    return {
        "output_dir": args.output_dir,
        "validation_grid": args.output_dir / "validation_grid.csv",
        "metrics_overall": args.output_dir / "metrics_overall.csv",
        "selection_comparison": args.output_dir / "selection_comparison.csv",
    }


def main() -> None:
    """Run the command-line entry point."""
    args = parse_args()
    run_conditional_knn_baseline_b(args)


if __name__ == "__main__":
    main()
