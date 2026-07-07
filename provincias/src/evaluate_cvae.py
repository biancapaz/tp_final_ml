"""
evaluate_cvae.py

Evaluation metrics for the CVAE:
  - Point error metrics (MAE, RMSE, MAPE) — comparable with Historical Mean
  - Generative metrics (coverage, sharpness, MMD, diversity)
  - Per-province breakdown

Works for both Architecture A (x_dim=528) and B (x_dim=24).
For Architecture B, includes reconstruction to system-level before metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from cvae import CVAE


PROVINCE_ORDER = [
    "buenos_aires", "catamarca", "chaco", "chubut", "cordoba",
    "corrientes", "entre_rios", "formosa", "jujuy", "la_pampa",
    "la_rioja", "mendoza", "misiones", "neuquen", "rio_negro",
    "salta", "san_juan", "san_luis", "santa_cruz", "santa_fe",
    "sgo_del_estero", "tucuman",
]
N_PROVINCES = len(PROVINCE_ORDER)
N_HOURS = 24


def extract_province_a(x: np.ndarray, province: str) -> np.ndarray:
    """Extract 24h profile for one province from Architecture A flat vector."""
    idx = PROVINCE_ORDER.index(province)
    return x[..., idx * N_HOURS : (idx + 1) * N_HOURS]


def generate_scenarios(
    model: CVAE,
    C: np.ndarray,
    n_scenarios: int = 500,
    batch_size: int = 64,
    device: str = "cpu",
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Generate n_scenarios demand profiles for each condition in C.

    temperature: scales the sampled z (see CVAE.generate). Select it on
                 validation only; report test with the chosen value fixed.
    Returns array of shape (len(C), n_scenarios, x_dim).
    """
    model.eval()
    dev = torch.device(device)
    model = model.to(dev)

    all_scenarios = []
    for i in range(0, len(C), batch_size):
        c_batch = torch.FloatTensor(C[i:i+batch_size]).to(dev)
        samples = model.generate(c_batch, n_samples=n_scenarios, temperature=temperature)
        # samples shape: (batch * n_scenarios, x_dim)
        samples = samples.cpu().numpy().reshape(len(c_batch), n_scenarios, -1)
        all_scenarios.append(samples)

    return np.concatenate(all_scenarios, axis=0)  # (n_days, n_scenarios, x_dim)


def point_metrics(
    X_real: np.ndarray,
    X_pred: np.ndarray,
    x_scaler,
) -> dict:
    """
    Compute point error metrics using the median of generated scenarios as prediction.

    X_real:  (n_days, x_dim) — real profiles in scaled space
    X_pred:  (n_days, n_scenarios, x_dim) or (n_days, x_dim) — generated scenarios
    x_scaler: fitted scaler for inverse transform
    """
    if X_pred.ndim == 3:
        X_pred_point = np.median(X_pred, axis=1)
    else:
        X_pred_point = X_pred

    real_mwh = x_scaler.inverse_transform(X_real)
    pred_mwh = x_scaler.inverse_transform(X_pred_point)

    mae_global  = np.abs(real_mwh - pred_mwh).mean()
    rmse_global = np.sqrt(((real_mwh - pred_mwh) ** 2).mean())

    energy_real = real_mwh.sum(axis=1)
    energy_pred = pred_mwh.sum(axis=1)
    mape_energy = (np.abs(energy_real - energy_pred) / energy_real).mean() * 100

    return {
        "mae_global":   mae_global,
        "rmse_global":  rmse_global,
        "mape_energy":  mape_energy,
    }


def generative_metrics(
    X_real: np.ndarray,
    X_scenarios: np.ndarray,
    x_scaler,
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
) -> dict:
    """
    Compute generative quality metrics.

    X_real:      (n_days, x_dim) in scaled space
    X_scenarios: (n_days, n_scenarios, x_dim) in scaled space

    Coverage: % of real profiles that fall within [P5, P95] band hour by hour.
    Sharpness: mean width of [P5, P95] band in MWh (lower = more informative).
    Diversity ratio: std of generated / std of real (should be ~1).
    MMD: maximum mean discrepancy between real and generated distributions.
    """
    real_mwh = x_scaler.inverse_transform(X_real)
    n_days, n_scen, x_dim = X_scenarios.shape
    scen_mwh = np.stack([
        x_scaler.inverse_transform(X_scenarios[:, s, :])
        for s in range(n_scen)
    ], axis=1)  # (n_days, n_scenarios, x_dim)

    p_low  = np.percentile(scen_mwh, percentile_low,  axis=1)  # (n_days, x_dim)
    p_high = np.percentile(scen_mwh, percentile_high, axis=1)

    # Coverage: real profile within band at each hour, averaged over days and hours
    within_band = (real_mwh >= p_low) & (real_mwh <= p_high)
    coverage = within_band.mean() * 100

    # Sharpness: mean band width
    sharpness = (p_high - p_low).mean()

    # Diversity ratio
    gen_std  = scen_mwh.std(axis=1).mean()
    real_std = real_mwh.std(axis=0).mean()
    diversity_ratio = gen_std / (real_std + 1e-8)

    # MMD (RBF kernel, subsample for speed)
    n_sub = min(500, n_days)
    idx   = np.random.choice(n_days, n_sub, replace=False)
    real_sub = real_mwh[idx]
    gen_sub  = scen_mwh[idx, np.random.randint(n_scen, size=n_sub), :]
    mmd = _mmd_rbf(real_sub, gen_sub)

    return {
        "coverage":       coverage,
        "sharpness":      sharpness,
        "diversity_ratio": diversity_ratio,
        "mmd":            mmd,
    }


def _mmd_rbf(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """Maximum Mean Discrepancy with RBF kernel."""
    def rbf(a, b):
        diff = a[:, None, :] - b[None, :, :]
        return np.exp(-np.sum(diff ** 2, axis=-1) / (2 * sigma ** 2))
    kxx = rbf(x, x).mean()
    kyy = rbf(y, y).mean()
    kxy = rbf(x, y).mean()
    return float(kxx + kyy - 2 * kxy)


def per_province_metrics_a(
    X_real: np.ndarray,
    X_scenarios: np.ndarray,
    x_scaler,
) -> pd.DataFrame:
    """
    Compute coverage and MAE per province for Architecture A.

    X_real:      (n_days, 528)
    X_scenarios: (n_days, n_scenarios, 528)
    """
    real_mwh = x_scaler.inverse_transform(X_real)
    n_days, n_scen, _ = X_scenarios.shape
    scen_mwh = np.stack([
        x_scaler.inverse_transform(X_scenarios[:, s, :])
        for s in range(n_scen)
    ], axis=1)

    rows = []
    for province in PROVINCE_ORDER:
        real_prov = extract_province_a(real_mwh, province)  # (n_days, 24)
        scen_prov = extract_province_a(scen_mwh, province)  # (n_days, n_scen, 24)

        median_pred = np.median(scen_prov, axis=1)
        mae = np.abs(real_prov - median_pred).mean()

        p5  = np.percentile(scen_prov, 5,  axis=1)
        p95 = np.percentile(scen_prov, 95, axis=1)
        coverage = ((real_prov >= p5) & (real_prov <= p95)).mean() * 100
        sharpness = (p95 - p5).mean()

        rows.append({
            "provincia":  province,
            "mae_mwh":    round(mae, 1),
            "coverage":   round(coverage, 1),
            "sharpness":  round(sharpness, 1),
        })

    return pd.DataFrame(rows).sort_values("mae_mwh", ascending=False).reset_index(drop=True)


def metrics_by_month_a(
    X_real: np.ndarray,
    X_scenarios: np.ndarray,
    dates: np.ndarray,
    x_scaler,
) -> pd.DataFrame:
    """
    Compute MAE, coverage and sharpness per calendar month for Architecture A.

    X_real:      (n_days, 528)
    X_scenarios: (n_days, n_scenarios, 528)
    dates:       (n_days,) array of dates matching X_real rows.
    """
    real_mwh = x_scaler.inverse_transform(X_real)
    n_days, n_scen, _ = X_scenarios.shape
    scen_mwh = np.stack([
        x_scaler.inverse_transform(X_scenarios[:, s, :])
        for s in range(n_scen)
    ], axis=1)

    median_pred = np.median(scen_mwh, axis=1)
    p5  = np.percentile(scen_mwh, 5,  axis=1)
    p95 = np.percentile(scen_mwh, 95, axis=1)

    months = pd.to_datetime(dates).month
    rows = []
    for mes in sorted(pd.unique(months)):
        mask = months == mes
        mae = np.abs(real_mwh[mask] - median_pred[mask]).mean()
        coverage = ((real_mwh[mask] >= p5[mask]) & (real_mwh[mask] <= p95[mask])).mean() * 100
        sharpness = (p95[mask] - p5[mask]).mean()
        rows.append({
            "mes":       int(mes),
            "n_dias":    int(mask.sum()),
            "mae_mwh":   round(mae, 1),
            "coverage":  round(coverage, 1),
            "sharpness": round(sharpness, 1),
        })
    return pd.DataFrame(rows).sort_values("mes").reset_index(drop=True)


def temperature_sweep(
    model: CVAE,
    X_val: np.ndarray,
    C_val: np.ndarray,
    x_scaler,
    temperatures: list[float],
    n_scenarios: int = 200,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Evaluate generative metrics on validation for each candidate temperature.

    Used to calibrate the latent sampling temperature without retraining:
    higher temperature widens generated bands (may raise coverage at the
    cost of sharpness), lower temperature narrows them. Must only be run
    on validation — never use test to pick a temperature.
    """
    rows = []
    for temperature in temperatures:
        scenarios = generate_scenarios(
            model, C_val, n_scenarios=n_scenarios, device=device, temperature=temperature,
        )
        pm = point_metrics(X_val, scenarios, x_scaler)
        gm = generative_metrics(X_val, scenarios, x_scaler)
        rows.append({
            "temperature":     temperature,
            "mae_global":      round(pm["mae_global"], 2),
            "rmse_global":     round(pm["rmse_global"], 2),
            "coverage":        round(gm["coverage"], 2),
            "sharpness":       round(gm["sharpness"], 2),
            "diversity_ratio": round(gm["diversity_ratio"], 4),
        })
    return pd.DataFrame(rows)


def select_temperature(sweep: pd.DataFrame, target_coverage: float = 90.0) -> float:
    """
    Select the temperature closest to the target coverage on validation.

    Tie-break order: coverage_gap (closer to target_coverage) first, then
    lower sharpness, then lower RMSE — the same generative criterion used
    for the Conditional KDE baseline, for consistency across models.
    """
    ranked = sweep.assign(
        coverage_gap=(sweep["coverage"] - target_coverage).abs()
    ).sort_values(["coverage_gap", "sharpness", "rmse_global"])
    return float(ranked.iloc[0]["temperature"])


def per_province_metrics_b(
    X_real: np.ndarray,
    X_scenarios: np.ndarray,
    provinces: np.ndarray,
    x_scaler,
) -> pd.DataFrame:
    """
    Compute coverage and MAE per province for Architecture B.

    X_real:      (n_samples, 24)  — one row per province-day
    X_scenarios: (n_samples, n_scenarios, 24)
    provinces:   (n_samples,) — province name for each row
    """
    real_mwh = x_scaler.inverse_transform(X_real)
    n_days, n_scen, _ = X_scenarios.shape
    scen_mwh = np.stack([
        x_scaler.inverse_transform(X_scenarios[:, s, :])
        for s in range(n_scen)
    ], axis=1)

    rows = []
    for province in PROVINCE_ORDER:
        mask = provinces == province
        if not mask.any():
            continue

        real_prov = real_mwh[mask]
        scen_prov = scen_mwh[mask]

        median_pred = np.median(scen_prov, axis=1)
        mae = np.abs(real_prov - median_pred).mean()

        p5  = np.percentile(scen_prov, 5,  axis=1)
        p95 = np.percentile(scen_prov, 95, axis=1)
        coverage = ((real_prov >= p5) & (real_prov <= p95)).mean() * 100
        sharpness = (p95 - p5).mean()

        rows.append({
            "provincia":  province,
            "mae_mwh":    round(mae, 1),
            "coverage":   round(coverage, 1),
            "sharpness":  round(sharpness, 1),
        })

    return pd.DataFrame(rows).sort_values("mae_mwh", ascending=False).reset_index(drop=True)


def posterior_prior_gap(
    model: CVAE,
    X: np.ndarray,
    C: np.ndarray,
    x_scaler,
    device: str = "cpu",
) -> dict:
    """
    Diagnose conditioning collapse by comparing two ways of decoding.

    Reconstructs each sample twice: once with z = encoder mu (the true
    posterior, which has seen the real x) and once with z sampled from the
    prior N(0,1) (what generation actually uses, since it only has c).

    A small gap means the decoder relies on c to reconstruct, as intended.
    A large gap means the decoder learned to depend on information leaked
    through z, so real scenario generation (prior-only) will be unreliable
    even though training reconstruction looks accurate.
    """
    model.eval()
    dev = torch.device(device)
    model = model.to(dev)

    with torch.no_grad():
        x_t = torch.FloatTensor(X).to(dev)
        c_t = torch.FloatTensor(C).to(dev)
        mu, _ = model.encoder(x_t, c_t)
        recon_posterior = model.decoder(mu, c_t).cpu().numpy()
        z_prior = torch.randn(len(c_t), model.latent_dim, device=dev)
        recon_prior = model.decoder(z_prior, c_t).cpu().numpy()

    real_mwh      = x_scaler.inverse_transform(X)
    posterior_mwh = x_scaler.inverse_transform(recon_posterior)
    prior_mwh     = x_scaler.inverse_transform(recon_prior)

    mae_posterior = float(np.abs(real_mwh - posterior_mwh).mean())
    mae_prior     = float(np.abs(real_mwh - prior_mwh).mean())

    return {
        "mae_posterior": mae_posterior,
        "mae_prior":     mae_prior,
        "gap_ratio":     mae_prior / mae_posterior if mae_posterior > 0 else float("inf"),
    }


def full_evaluation(
    model: CVAE,
    X_real: np.ndarray,
    C: np.ndarray,
    x_scaler,
    n_scenarios: int = 500,
    device: str = "cpu",
    provinces: np.ndarray | None = None,
    architecture: str = "A",
    temperature: float = 1.0,
) -> dict:
    """
    Run full evaluation: generate scenarios and compute all metrics.

    architecture: 'A' for system-level, 'B' for province-day.
    provinces: required for architecture B.
    temperature: latent sampling temperature (see CVAE.generate). Only
                 change this from 1.0 using a value selected on validation.
    """
    print(f"Generando {n_scenarios} escenarios para {len(X_real)} muestras...")
    X_scenarios = generate_scenarios(
        model, C, n_scenarios=n_scenarios, device=device, temperature=temperature,
    )

    print("Calculando metricas...")
    metrics = {}
    metrics.update(point_metrics(X_real, X_scenarios, x_scaler))
    metrics.update(generative_metrics(X_real, X_scenarios, x_scaler))

    if architecture == "A":
        metrics["per_province"] = per_province_metrics_a(X_real, X_scenarios, x_scaler)
    else:
        if provinces is None:
            raise ValueError("provinces required for architecture B")
        metrics["per_province"] = per_province_metrics_b(X_real, X_scenarios, provinces, x_scaler)

    return metrics, X_scenarios
