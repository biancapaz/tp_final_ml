"""
preprocessing_b.py

Builds dataset for Architecture B (province-day format).
Each sample is one day for one province:
  x = 24-hour demand profile of a single province
  c = climate of that province + calendar + socioeconomic indicators

This allows the CVAE to learn a general rule:
  high population + industry -> high demand
using socioeconomic indicators as real varying features across samples.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROVINCE_ORDER = [
    "buenos_aires", "catamarca", "chaco", "chubut", "cordoba",
    "corrientes", "entre_rios", "formosa", "jujuy", "la_pampa",
    "la_rioja", "mendoza", "misiones", "neuquen", "rio_negro",
    "salta", "san_juan", "san_luis", "santa_cruz", "santa_fe",
    "sgo_del_estero", "tucuman",
]
N_PROVINCES = len(PROVINCE_ORDER)
N_HOURS = 24

TRAIN_YEARS = [2019, 2021, 2022, 2023]
VAL_YEARS   = [2024]
TEST_YEARS  = [2025, 2026]

CLIMATE_FEATURES = [
    "temp_max", "temp_min", "temp_media", "temp_amplitud",
    "sensacion_max", "sensacion_min", "sensacion_media", "sensacion_amplitud",
    "humedad_prom", "viento_prom", "grado_calor", "grado_frio",
]
CALENDAR_FEATURES  = ["mes", "dia_semana", "es_feriado"]
SOCIOECO_FEATURES = ["poblacion", "pbg_per_capita", "participacion_industrial", "superficie_km2"]

def build_x_long(x_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Convert x from wide format (1 row per day, 528 cols) to long format
    (1 row per day-province, 24 cols).

    x_wide: DataFrame with columns fecha, {province}_{hour:02d}, ...
    Returns: DataFrame with columns fecha, provincia, hora_01, ..., hora_24
    """
    records = []
    for provincia in PROVINCE_ORDER:
        cols = [f"{provincia}_{h:02d}" for h in range(1, 25)]
        if not all(c in x_wide.columns for c in cols):
            print(f"  Warning: provincia {provincia} not found in x_wide")
            continue
        df_prov = x_wide[["fecha"] + cols].copy()
        df_prov.columns = ["fecha"] + [f"hora_{h:02d}" for h in range(1, 25)]
        df_prov["provincia"] = provincia
        records.append(df_prov)

    df_long = pd.concat(records, ignore_index=True)
    df_long = df_long.sort_values(["fecha", "provincia"]).reset_index(drop=True)
    return df_long


def build_c_long(
    clima_diario: pd.DataFrame,
    calendario: pd.DataFrame,
    indicadores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build condition matrix in long format (1 row per day-province).

    clima_diario:  DataFrame with columns fecha, provincia, + CLIMATE_FEATURES
    calendario:    DataFrame with columns fecha, mes, dia_semana, es_feriado
    indicadores:   DataFrame with columns provincia, + SOCIOECO_FEATURES

    Returns: DataFrame with columns fecha, provincia, + all condition features
    """
    # Start from climate (already long format)
    c = clima_diario[["fecha", "provincia"] + CLIMATE_FEATURES].copy()

    # Join calendar (same for all provinces on the same day)
    c = c.merge(calendario[["fecha"] + CALENDAR_FEATURES], on="fecha", how="left")

    # Join socioeconomic indicators (same for each province across all days)
    c = c.merge(indicadores[["provincia"] + SOCIOECO_FEATURES], on="provincia", how="left")

    c = c.sort_values(["fecha", "provincia"]).reset_index(drop=True)
    return c


def temporal_split_long(
    df: pd.DataFrame,
    date_col: str = "fecha",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split long-format dataframe by year. Keeps province-day pairs together.
    2020 excluded automatically (not in TRAIN/VAL/TEST_YEARS).
    """
    years = df[date_col].dt.year
    train = df[years.isin(TRAIN_YEARS)].reset_index(drop=True).copy()
    val   = df[years.isin(VAL_YEARS)].reset_index(drop=True).copy()
    test  = df[years.isin(TEST_YEARS)].reset_index(drop=True).copy()
    return train, val, test


def fit_x_scaler_b(x_train: pd.DataFrame) -> StandardScaler:
    """
    Fit StandardScaler on training demand profiles (Architecture B).
    Each row is a 24-hour profile of one province.
    Fitted exclusively on train.
    """
    hora_cols = [f"hora_{h:02d}" for h in range(1, 25)]
    scaler = StandardScaler()
    scaler.fit(x_train[hora_cols].values)
    return scaler


def transform_x_b(x: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Apply pre-fitted scaler to x in long format. Returns array (n_samples, 24)."""
    hora_cols = [f"hora_{h:02d}" for h in range(1, 25)]
    return scaler.transform(x[hora_cols].values)


def fit_c_scaler_b(c_train: pd.DataFrame) -> tuple[StandardScaler, list[str], list[str]]:
    """
    Fit StandardScaler on continuous condition features (Architecture B).
    Separates binary (es_feriado) from continuous features.
    Fitted exclusively on train.
    Returns: scaler, cols_to_scale, binary_cols
    """
    all_feature_cols = CLIMATE_FEATURES + ["mes", "dia_semana"] + SOCIOECO_FEATURES
    binary_cols = ["es_feriado"]

    cols_to_scale = [c for c in all_feature_cols if c in c_train.columns]
    binary_cols   = [c for c in binary_cols if c in c_train.columns]

    scaler = StandardScaler()
    scaler.fit(c_train[cols_to_scale].values)
    return scaler, cols_to_scale, binary_cols


def transform_c_b(
    c: pd.DataFrame,
    scaler: StandardScaler,
    cols_to_scale: list[str],
    binary_cols: list[str],
) -> np.ndarray:
    """Apply pre-fitted scaler to c. Returns array (n_samples, c_dim)."""
    scaled = scaler.transform(c[cols_to_scale].values)
    binary = c[binary_cols].values
    return np.concatenate([binary, scaled], axis=1)


def run_preprocessing_b(
    processed_dir: Path,
    save_dir: Path | None = None,
) -> dict:
    """
    Full preprocessing pipeline for Architecture B (province-day format).

    1. Load wide x, climate, calendar, indicators.
    2. Build long-format x and c.
    3. Remove anomalous days.
    4. Split by year.
    5. Fit scalers on train only, transform all splits.
    6. Save to disk.
    """
    # Load
    x_wide     = pd.read_csv(processed_dir / "x_perfiles_diarios_provincias.csv", parse_dates=["fecha"])
    clima      = pd.read_csv(processed_dir / "clima_diario_por_provincia.csv",      parse_dates=["fecha"])
    calendario = pd.read_csv(processed_dir / "c_calendario.csv",                    parse_dates=["fecha"])
    indicadores = pd.read_csv(processed_dir / "indicadores_provinciales.csv")

    print(f"x_wide:      {x_wide.shape}")
    print(f"clima:       {clima.shape}")
    print(f"calendario:  {calendario.shape}")
    print(f"indicadores: {indicadores.shape}")

    # Build long format
    x_long = build_x_long(x_wide)
    c_long = build_c_long(clima, calendario, indicadores)

    print(f"\nx_long: {x_long.shape}  ({len(x_long)} province-day samples)")
    print(f"c_long: {c_long.shape}")

    # Align dates and provinces
    common_keys = set(zip(x_long["fecha"], x_long["provincia"])) & \
                  set(zip(c_long["fecha"], c_long["provincia"]))
    x_long = x_long[
        pd.Series(list(zip(x_long["fecha"], x_long["provincia"]))).isin(common_keys).values
    ].reset_index(drop=True)
    c_long = c_long[
        pd.Series(list(zip(c_long["fecha"], c_long["provincia"]))).isin(common_keys).values
    ].reset_index(drop=True)

    # Remove anomalous days (same dates as Architecture A)
    anomaly_dates = pd.to_datetime(["2023-12-17", "2023-03-01", "2019-06-16"])
    x_long = x_long[~x_long["fecha"].isin(anomaly_dates)].reset_index(drop=True)
    c_long = c_long[~c_long["fecha"].isin(anomaly_dates)].reset_index(drop=True)
    print(f"\nDespues de eliminar dias anomalos: {len(x_long)} samples")

    # Split
    x_train, x_val, x_test = temporal_split_long(x_long)
    c_train, c_val, c_test = temporal_split_long(c_long)

    print(f"\nSplit:")
    print(f"  train: {len(x_train)} samples ({len(x_train)//N_PROVINCES} dias × {N_PROVINCES} provincias)")
    print(f"  val:   {len(x_val)} samples")
    print(f"  test:  {len(x_test)} samples")

    # Fit scalers on train only
    x_scaler = fit_x_scaler_b(x_train)
    c_scaler, cols_to_scale, binary_cols = fit_c_scaler_b(c_train)

    # Transform all splits
    X_train = transform_x_b(x_train, x_scaler)
    X_val   = transform_x_b(x_val,   x_scaler)
    X_test  = transform_x_b(x_test,  x_scaler)

    C_train = transform_c_b(c_train, c_scaler, cols_to_scale, binary_cols)
    C_val   = transform_c_b(c_val,   c_scaler, cols_to_scale, binary_cols)
    C_test  = transform_c_b(c_test,  c_scaler, cols_to_scale, binary_cols)

    print(f"\nShapes finales:")
    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    print(f"  C_train: {C_train.shape}, C_val: {C_val.shape}, C_test: {C_test.shape}")
    print(f"  x_dim: {X_train.shape[1]}, c_dim: {C_train.shape[1]}")

    result = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "C_train": C_train, "C_val": C_val, "C_test": C_test,
        "dates_train":     x_train["fecha"].values,
        "dates_val":       x_val["fecha"].values,
        "dates_test":      x_test["fecha"].values,
        "provinces_train": x_train["provincia"].values,
        "provinces_val":   x_val["provincia"].values,
        "provinces_test":  x_test["provincia"].values,
        "x_scaler":        x_scaler,
        "c_scaler":        c_scaler,
        "cols_to_scale":   cols_to_scale,
        "binary_cols":     binary_cols,
        "x_dim":           X_train.shape[1],
        "c_dim":           C_train.shape[1],
    }

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for name in ["X_train", "X_val", "X_test", "C_train", "C_val", "C_test"]:
            np.save(save_dir / f"b_{name}.npy", result[name])

        for name in ["dates_train", "dates_val", "dates_test",
                     "provinces_train", "provinces_val", "provinces_test"]:
            np.save(save_dir / f"b_{name}.npy", result[name])

        with open(save_dir / "b_x_scaler.pkl", "wb") as f:
            pickle.dump(x_scaler, f)
        with open(save_dir / "b_c_scaler.pkl", "wb") as f:
            pickle.dump(c_scaler, f)
        with open(save_dir / "b_meta.pkl", "wb") as f:
            pickle.dump({"cols_to_scale": cols_to_scale, "binary_cols": binary_cols,
                         "x_dim": X_train.shape[1], "c_dim": C_train.shape[1]}, f)

        print(f"\nArchivos guardados en {save_dir}/ con prefijo 'b_'")

    return result
