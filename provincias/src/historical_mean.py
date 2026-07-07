"""
historical_mean.py

Historical Mean Profile baseline for the provincial electricity demand model.
For each (mes, dia_semana, es_feriado) group, computes the mean demand profile
over all training days in that group.

This is the minimum baseline — no parameters, no learning.
Any model that fails to outperform this in MAE is not working correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


PROVINCE_ORDER = [
    "buenos_aires", "catamarca", "chaco", "chubut", "cordoba",
    "corrientes", "entre_rios", "formosa", "jujuy", "la_pampa",
    "la_rioja", "mendoza", "misiones", "neuquen", "rio_negro",
    "salta", "san_juan", "san_luis", "santa_cruz", "santa_fe",
    "sgo_del_estero", "tucuman",
]
N_PROVINCES = len(PROVINCE_ORDER)
N_HOURS = 24
X_DIM = N_PROVINCES * N_HOURS  # 528


def extract_province(x: np.ndarray, province: str) -> np.ndarray:
    """Extract the 24-hour demand profile for a single province from a flattened x vector."""
    idx = PROVINCE_ORDER.index(province)
    return x[..., idx * N_HOURS : (idx + 1) * N_HOURS]


# ─────────────────────────────────────────────
# 1. Fit
# ─────────────────────────────────────────────

MIN_DAYS_PER_GROUP = 3  # minimum days to trust a group mean


def fit(
    X_train: np.ndarray,
    dates_train: np.ndarray,
    c_calendar: pd.DataFrame,
    min_days: int = MIN_DAYS_PER_GROUP,
) -> dict:
    """
    Compute mean demand profiles using a hierarchical fallback strategy.
    Groups with fewer than min_days samples fall back to a broader group.

    Fallback hierarchy:
      (mes, dia_semana, es_feriado)  — most specific
      (mes, dia_semana)              — ignores holiday flag
      (mes)                          — only month
      global mean                    — final fallback

    X_train:     scaled demand profiles, shape (n_train, 528).
    dates_train: array of dates corresponding to X_train rows.
    c_calendar:  DataFrame with columns fecha, mes, dia_semana, es_feriado.
                 Must use original (unscaled) calendar values.
    min_days:    minimum number of training days required to trust a group mean.

    Returns a dict with profiles at each level and group counts.
    """
    dates_df = pd.DataFrame({"fecha": pd.to_datetime(dates_train)})
    dates_df = dates_df.merge(
        c_calendar[["fecha", "mes", "dia_semana", "es_feriado"]],
        on="fecha",
        how="left",
    )

    def build_level(group_keys: list[str]) -> tuple[dict, dict]:
        profiles = {}
        counts = {}
        for key, idx in dates_df.groupby(group_keys).groups.items():
            if len(idx) >= min_days:
                profiles[key] = X_train[idx.values].mean(axis=0)
                counts[key] = len(idx)
        return profiles, counts

    level3, counts3 = build_level(["mes", "dia_semana", "es_feriado"])
    level2, counts2 = build_level(["mes", "dia_semana"])
    level1, counts1 = build_level(["mes"])
    global_mean = X_train.mean(axis=0)

    return {
        "level3":      level3,
        "level2":      level2,
        "level1":      level1,
        "global_mean": global_mean,
        "counts3":     counts3,
        "counts2":     counts2,
        "counts1":     counts1,
        "min_days":    min_days,
    }


# ─────────────────────────────────────────────
# 2. Predict
# ─────────────────────────────────────────────

def predict(
    dates: np.ndarray,
    c_calendar: pd.DataFrame,
    model: dict,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Predict demand profiles using the hierarchical fallback strategy.

    For each day, tries groups in order:
      1. (mes, dia_semana, es_feriado)
      2. (mes, dia_semana)
      3. (mes)
      4. global mean

    dates:      array of dates to predict.
    c_calendar: DataFrame with columns fecha, mes, dia_semana, es_feriado.
    model:      dict returned by fit().

    Returns:
      predictions: numpy array of shape (n_days, 528).
      fallback_log: DataFrame showing which level was used for each day.
    """
    dates_df = pd.DataFrame({"fecha": pd.to_datetime(dates)})
    dates_df = dates_df.merge(
        c_calendar[["fecha", "mes", "dia_semana", "es_feriado"]],
        on="fecha",
        how="left",
    )

    level3      = model["level3"]
    level2      = model["level2"]
    level1      = model["level1"]
    global_mean = model["global_mean"]

    predictions  = np.zeros((len(dates_df), X_DIM))
    fallback_log = []

    for i, row in dates_df.iterrows():
        mes        = int(row["mes"])
        dia_semana = int(row["dia_semana"])
        es_feriado = int(row["es_feriado"])
        fecha      = row["fecha"]

        key3 = (mes, dia_semana, es_feriado)
        key2 = (mes, dia_semana)
        key1 = (mes,)

        if key3 in level3:
            predictions[i] = level3[key3]
            level_used = "level3"
        elif key2 in level2:
            predictions[i] = level2[key2]
            level_used = "level2"
        elif key1 in level1:
            predictions[i] = level1[key1]
            level_used = "level3"
        else:
            predictions[i] = global_mean
            level_used = "global"

        if level_used != "level3":
            fallback_log.append({
                "fecha":      fecha,
                "mes":        mes,
                "dia_semana": dia_semana,
                "es_feriado": es_feriado,
                "level_used": level_used,
            })

    fallback_df = pd.DataFrame(fallback_log) if fallback_log else pd.DataFrame(
        columns=["fecha", "mes", "dia_semana", "es_feriado", "level_used"]
    )

    return predictions, fallback_df


# ─────────────────────────────────────────────
# 3. Evaluate
# ─────────────────────────────────────────────

def evaluate(
    X_real: np.ndarray,
    X_pred: np.ndarray,
    dates: np.ndarray,
    x_scaler,
) -> dict:
    """
    Evaluate predictions against real profiles.
    Both X_real and X_pred must be in scaled space — this function
    inverse-transforms them before computing metrics.

    Returns a dict with global and per-province metrics.
    """
    # Inverse transform to MWh
    X_real_mwh = x_scaler.inverse_transform(X_real)
    X_pred_mwh = x_scaler.inverse_transform(X_pred)

    # Global MAE horario
    mae_global = np.abs(X_real_mwh - X_pred_mwh).mean()
    rmse_global = np.sqrt(((X_real_mwh - X_pred_mwh) ** 2).mean())

    # Energy MAPE — energia diaria total
    energy_real = X_real_mwh.sum(axis=1)
    energy_pred = X_pred_mwh.sum(axis=1)
    mape_energy = (np.abs(energy_real - energy_pred) / energy_real).mean() * 100

    # MAE por provincia
    mae_by_province = {}
    rmse_by_province = {}
    mape_by_province = {}

    for province in PROVINCE_ORDER:
        real_prov = extract_province(X_real_mwh, province)
        pred_prov = extract_province(X_pred_mwh, province)

        mae_by_province[province]  = np.abs(real_prov - pred_prov).mean()
        rmse_by_province[province] = np.sqrt(((real_prov - pred_prov) ** 2).mean())

        energy_real_prov = real_prov.sum(axis=1)
        energy_pred_prov = pred_prov.sum(axis=1)
        mape_by_province[province] = (
            np.abs(energy_real_prov - energy_pred_prov) / energy_real_prov
        ).mean() * 100

    return {
        "mae_global":       mae_global,
        "rmse_global":      rmse_global,
        "mape_energy":      mape_energy,
        "mae_by_province":  mae_by_province,
        "rmse_by_province": rmse_by_province,
        "mape_by_province": mape_by_province,
        "n_days":           len(X_real),
    }


def evaluation_summary(metrics: dict) -> pd.DataFrame:
    """
    Format per-province metrics as a sorted DataFrame for display.
    """
    rows = []
    for province in PROVINCE_ORDER:
        rows.append({
            "provincia":  province,
            "mae_mwh":    round(metrics["mae_by_province"][province], 1),
            "rmse_mwh":   round(metrics["rmse_by_province"][province], 1),
            "mape_pct":   round(metrics["mape_by_province"][province], 2),
        })
    df = pd.DataFrame(rows).sort_values("mae_mwh", ascending=False).reset_index(drop=True)
    return df


def evaluate_by_month(
    X_real: np.ndarray,
    X_pred: np.ndarray,
    dates: np.ndarray,
    c_calendar: pd.DataFrame,
    x_scaler,
) -> pd.DataFrame:
    """
    Evaluate predictions grouped by calendar month.

    Both X_real and X_pred must be in scaled space — this function
    inverse-transforms them before computing metrics. Returns a DataFrame
    with one row per month (1-12) and MAE, RMSE and energy MAPE.
    """
    X_real_mwh = x_scaler.inverse_transform(X_real)
    X_pred_mwh = x_scaler.inverse_transform(X_pred)

    dates_df = pd.DataFrame({"fecha": pd.to_datetime(dates)}).merge(
        c_calendar[["fecha", "mes"]], on="fecha", how="left"
    )

    rows = []
    for mes, idx in dates_df.groupby("mes").groups.items():
        real_month = X_real_mwh[idx.values]
        pred_month = X_pred_mwh[idx.values]
        energy_real = real_month.sum(axis=1)
        energy_pred = pred_month.sum(axis=1)
        rows.append({
            "mes":        int(mes),
            "n_dias":     len(idx),
            "mae_mwh":    round(np.abs(real_month - pred_month).mean(), 1),
            "rmse_mwh":   round(np.sqrt(((real_month - pred_month) ** 2).mean()), 1),
            "mape_pct":   round((np.abs(energy_real - energy_pred) / energy_real).mean() * 100, 2),
        })
    return pd.DataFrame(rows).sort_values("mes").reset_index(drop=True)