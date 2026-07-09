"""
Executable final model for the Machine Learning final project.

The selected final model is CVAE-A. It generates probabilistic hourly demand
scenarios for the complete provincial system:

    22 provinces x 24 hours = 528 values per day.

Expected input format: a CSV with one row per day, a ``fecha`` column, and the
same CVAE-A condition columns used during development. If the CSV also includes
the 528 demand columns, the script computes test metrics automatically.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
PROVINCIAS_SRC = PROJECT_ROOT / "provincias" / "src"
if str(PROVINCIAS_SRC) not in sys.path:
    sys.path.insert(0, str(PROVINCIAS_SRC))

from cvae import CVAE  # noqa: E402
from preprocessing import transform_c as _transform_c  # noqa: E402

try:
    from evaluate_cvae import _mmd_rbf, generate_scenarios as _generate_scenarios  # noqa: E402
except Exception:  # pragma: no cover - optional metric helper
    _mmd_rbf = None
    _generate_scenarios = None


DATE_COLUMN = "fecha"


def set_seed(seed: int = 42) -> None:
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {path}. Run the artifact export step or check "
            f"--artifacts-dir."
        )


def validate_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Device 'cuda' was requested, but CUDA is not available.")
    if device == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("Device 'mps' was requested, but MPS is not available.")
    return torch.device(device)


def load_artifacts(artifacts_dir: Path, device: str = "cpu") -> dict[str, Any]:
    artifacts_dir = artifacts_dir.resolve()
    config_path = artifacts_dir / "cvae_a_config.json"
    metadata_path = artifacts_dir / "cvae_a_metadata.json"
    checkpoint_path = artifacts_dir / "cvae_a_final.pt"
    x_scaler_path = artifacts_dir / "x_scaler_a.pkl"
    c_scaler_path = artifacts_dir / "c_scaler_a.pkl"

    for path, label in [
        (config_path, "CVAE-A config"),
        (metadata_path, "CVAE-A metadata"),
        (checkpoint_path, "CVAE-A checkpoint"),
        (x_scaler_path, "x scaler"),
        (c_scaler_path, "c scaler"),
    ]:
        _require_file(path, label)

    config = _load_json(config_path)
    metadata = _load_json(metadata_path)
    checkpoint = _torch_load(checkpoint_path)
    x_scaler = _load_pickle(x_scaler_path)
    c_scaler = _load_pickle(c_scaler_path)

    model = CVAE(
        x_dim=int(config["x_dim"]),
        c_dim=int(config["c_dim"]),
        hidden_dims=list(config["hidden_dims"]),
        latent_dim=int(config["latent_dim"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(validate_device(device))
    model.eval()

    return {
        "artifacts_dir": artifacts_dir,
        "paths": {
            "config": config_path,
            "metadata": metadata_path,
            "checkpoint": checkpoint_path,
            "x_scaler": x_scaler_path,
            "c_scaler": c_scaler_path,
        },
        "config": config,
        "metadata": metadata,
        "model": model,
        "x_scaler": x_scaler,
        "c_scaler": c_scaler,
    }


def prepare_input(input_path: Path, artifacts: dict[str, Any]) -> dict[str, Any]:
    metadata = artifacts["metadata"]
    c_scaler = artifacts["c_scaler"]
    x_scaler = artifacts["x_scaler"]

    _require_file(input_path, "input CSV")
    df = pd.read_csv(input_path)
    if DATE_COLUMN not in df.columns:
        raise ValueError(f"Input CSV must include a '{DATE_COLUMN}' column.")

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    c_groups = metadata["c_column_groups"]
    binary_cols = c_groups["calendar_binary"]
    cols_to_scale = (
        c_groups["calendar_ordinal"]
        + c_groups["climate"]
        + c_groups["socioeconomic"]
    )
    required_condition_cols = binary_cols + cols_to_scale
    missing_condition_cols = [
        col for col in required_condition_cols if col not in df.columns
    ]
    if missing_condition_cols:
        preview = ", ".join(missing_condition_cols[:12])
        more = "" if len(missing_condition_cols) <= 12 else " ..."
        raise ValueError(
            "Input CSV is missing CVAE-A condition columns. "
            f"Missing {len(missing_condition_cols)} columns: {preview}{more}. "
            "Use the same wide format as provincias/data/processed/"
            "c_condiciones_diarias.csv, optionally merged with demand columns."
        )

    C = _transform_c(df, c_scaler, c_groups).astype(np.float32)

    expected_c_dim = int(artifacts["config"]["c_dim"])
    if C.shape[1] != expected_c_dim:
        raise ValueError(
            f"Prepared C has {C.shape[1]} columns, expected {expected_c_dim}."
        )

    demand_columns = metadata["demand_columns"]
    has_real_demand = all(col in df.columns for col in demand_columns)
    X_real = None
    if has_real_demand:
        X_real = x_scaler.transform(df[demand_columns].to_numpy(dtype=np.float32))
        X_real = X_real.astype(np.float32)

    return {
        "dates": df[DATE_COLUMN].dt.strftime("%Y-%m-%d").to_numpy(),
        "C": C,
        "X_real": X_real,
        "has_real_demand": has_real_demand,
        "n_days": len(df),
    }


def generate_samples(
    model: CVAE,
    C: np.ndarray,
    n_samples: int = 500,
    device: str = "cpu",
    temperature: float = 1.0,
    batch_size: int = 64,
) -> np.ndarray:
    dev = validate_device(device)
    if _generate_scenarios is not None:
        return _generate_scenarios(
            model,
            C,
            n_scenarios=n_samples,
            batch_size=batch_size,
            device=device,
            temperature=temperature,
        ).astype(np.float32)

    model = model.to(dev)
    model.eval()

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(C), batch_size):
            c_batch = torch.as_tensor(C[start : start + batch_size], dtype=torch.float32)
            c_batch = c_batch.to(dev)
            generated = model.generate(
                c_batch,
                n_samples=n_samples,
                temperature=temperature,
            )
            batch = generated.detach().cpu().numpy()
            batch = batch.reshape(len(c_batch), n_samples, -1)
            chunks.append(batch)

    return np.concatenate(chunks, axis=0).astype(np.float32)


def inverse_transform_outputs(
    samples_scaled: np.ndarray,
    x_scaler: Any,
    n_provinces: int,
    n_hours: int,
) -> np.ndarray:
    n_days, n_samples, x_dim = samples_scaled.shape
    expected_dim = n_provinces * n_hours
    if x_dim != expected_dim:
        raise ValueError(f"Generated x_dim={x_dim}, expected {expected_dim}.")

    flat_scaled = samples_scaled.reshape(-1, x_dim)
    flat_mwh = x_scaler.inverse_transform(flat_scaled)
    return flat_mwh.reshape(n_days, n_samples, n_provinces, n_hours)


def _prediction_frame(
    values: np.ndarray,
    dates: np.ndarray,
    provinces: list[str],
    hours: list[int],
    value_column: str,
) -> pd.DataFrame:
    if values.shape != (len(dates), len(provinces), len(hours)):
        raise ValueError(
            f"Unexpected prediction shape {values.shape}; expected "
            f"({len(dates)}, {len(provinces)}, {len(hours)})."
        )

    return pd.DataFrame(
        {
            "fecha": np.repeat(dates, len(provinces) * len(hours)),
            "provincia": np.tile(np.repeat(provinces, len(hours)), len(dates)),
            "hora": np.tile(hours, len(dates) * len(provinces)),
            value_column: values.reshape(-1),
        }
    )


def save_predictions(
    output_dir: Path,
    dates: np.ndarray,
    samples_mwh: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    provinces = metadata["province_order"]
    hours = metadata["hours"]

    median = np.median(samples_mwh, axis=1)
    p05 = np.percentile(samples_mwh, 5, axis=1)
    p95 = np.percentile(samples_mwh, 95, axis=1)

    paths = {
        "median": output_dir / "cvae_a_median.csv",
        "p05": output_dir / "cvae_a_p05.csv",
        "p95": output_dir / "cvae_a_p95.csv",
        "samples": output_dir / "cvae_a_samples.npy",
    }

    _prediction_frame(median, dates, provinces, hours, "median_mwh").to_csv(
        paths["median"], index=False
    )
    _prediction_frame(p05, dates, provinces, hours, "p05_mwh").to_csv(
        paths["p05"], index=False
    )
    _prediction_frame(p95, dates, provinces, hours, "p95_mwh").to_csv(
        paths["p95"], index=False
    )
    np.save(paths["samples"], samples_mwh.astype(np.float32))

    return paths


def _inverse_real(X_real: np.ndarray, x_scaler: Any, metadata: dict[str, Any]) -> np.ndarray:
    n_provinces = int(metadata["n_provinces"])
    n_hours = int(metadata["n_hours"])
    real_mwh = x_scaler.inverse_transform(X_real)
    return real_mwh.reshape(len(X_real), n_provinces, n_hours)


def compute_metrics_if_available(
    X_real: np.ndarray | None,
    samples_mwh: np.ndarray,
    x_scaler: Any,
    metadata: dict[str, Any],
    output_dir: Path,
) -> dict[str, float | None] | None:
    if X_real is None:
        return None

    real_mwh = _inverse_real(X_real, x_scaler, metadata)
    median = np.median(samples_mwh, axis=1)
    p05 = np.percentile(samples_mwh, 5, axis=1)
    p95 = np.percentile(samples_mwh, 95, axis=1)

    coverage = ((real_mwh >= p05) & (real_mwh <= p95)).mean() * 100.0
    sharpness = (p95 - p05).mean()
    mae_median = np.abs(real_mwh - median).mean()
    rmse_mean_profile = np.sqrt(((real_mwh - median) ** 2).mean())

    energy_real = real_mwh.sum(axis=(1, 2))
    energy_pred = median.sum(axis=(1, 2))
    mape_energy = (
        np.abs(energy_real - energy_pred) / np.maximum(energy_real, 1e-8)
    ).mean() * 100.0

    mmd = None
    if _mmd_rbf is not None:
        real_flat = real_mwh.reshape(len(real_mwh), -1)
        scen_flat = samples_mwh.reshape(samples_mwh.shape[0], samples_mwh.shape[1], -1)
        n_days, n_scenarios, _ = scen_flat.shape
        n_sub = min(500, n_days)
        idx = np.random.choice(n_days, n_sub, replace=False)
        scenario_idx = np.random.randint(n_scenarios, size=n_sub)
        mmd = float(_mmd_rbf(real_flat[idx], scen_flat[idx, scenario_idx, :]))

    metrics: dict[str, float | None] = {
        "coverage_p5_p95": float(coverage),
        "sharpness_mwh": float(sharpness),
        "rmse_mean_profile_mwh": float(rmse_mean_profile),
        "mae_median_mwh": float(mae_median),
        "mmd": mmd,
        "mape_energy": float(mape_energy),
    }

    with (output_dir / "metrics_test.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")

    return metrics


def save_run_metadata(
    output_dir: Path,
    artifacts: dict[str, Any],
    seed: int,
    n_samples: int,
    device: str,
    n_days: int,
    input_path: Path,
    output_paths: dict[str, Path],
    metrics_written: bool,
) -> Path:
    config = artifacts["config"]
    metadata = {
        "model": "CVAE-A",
        "seed": seed,
        "n_samples": n_samples,
        "device": device,
        "run_datetime": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "output_files": {key: str(path) for key, path in output_paths.items()},
        "artifacts_dir": str(artifacts["artifacts_dir"]),
        "artifact_paths": {
            key: str(path) for key, path in artifacts["paths"].items()
        },
        "dimensions": {
            "x_dim": int(config["x_dim"]),
            "c_dim": int(config["c_dim"]),
            "latent_dim": int(config["latent_dim"]),
            "n_provinces": int(artifacts["metadata"]["n_provinces"]),
            "n_hours": int(artifacts["metadata"]["n_hours"]),
        },
        "temperature": float(config.get("temperature", 1.0)),
        "n_days_processed": int(n_days),
        "metrics_test_written": metrics_written,
    }
    path = output_dir / "run_metadata.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run final CVAE-A model and save generated demand scenarios."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input CSV path.")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for predictions and metrics.",
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument(
        "--n-samples",
        "--n_samples",
        dest="n_samples",
        default=500,
        type=int,
        help="Number of scenarios generated per day.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device. Default: cpu.")
    parser.add_argument(
        "--artifacts-dir",
        default=Path("models/final_cvae_a"),
        type=Path,
        help="Directory with final CVAE-A artifacts.",
    )
    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        help="Generation batch size.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be a positive integer.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")

    set_seed(args.seed)

    artifacts = load_artifacts(args.artifacts_dir, device=args.device)
    prepared = prepare_input(args.input, artifacts)

    temperature = float(artifacts["config"].get("temperature", 1.0))
    samples_scaled = generate_samples(
        artifacts["model"],
        prepared["C"],
        n_samples=args.n_samples,
        device=args.device,
        temperature=temperature,
        batch_size=args.batch_size,
    )
    samples_mwh = inverse_transform_outputs(
        samples_scaled,
        artifacts["x_scaler"],
        int(artifacts["metadata"]["n_provinces"]),
        int(artifacts["metadata"]["n_hours"]),
    )

    output_paths = save_predictions(
        args.output,
        prepared["dates"],
        samples_mwh,
        artifacts["metadata"],
    )
    metrics = compute_metrics_if_available(
        prepared["X_real"],
        samples_mwh,
        artifacts["x_scaler"],
        artifacts["metadata"],
        args.output,
    )
    save_run_metadata(
        args.output,
        artifacts,
        args.seed,
        args.n_samples,
        args.device,
        prepared["n_days"],
        args.input,
        output_paths,
        metrics_written=metrics is not None,
    )

    print("Final model: CVAE-A")
    print(f"Seed: {args.seed}")
    print(f"n_samples: {args.n_samples}")
    print(f"Device: {args.device}")
    print("")
    if metrics is not None:
        print(f"Coverage P5-P95: {metrics['coverage_p5_p95']:.2f}%")
        print(f"Sharpness: {metrics['sharpness_mwh']:.2f} MWh")
        print(f"MAE median: {metrics['mae_median_mwh']:.2f} MWh")
        print(f"RMSE mean profile: {metrics['rmse_mean_profile_mwh']:.2f} MWh")
        mmd = metrics["mmd"]
        print(f"MMD: {mmd:.6f}" if mmd is not None else "MMD: null")
    else:
        print("Input has no real demand columns; metrics_test.json was not written.")
    print(f"Outputs saved to: {args.output}")


if __name__ == "__main__":
    main()
