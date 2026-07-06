"""
preprocessing.py

Funciones de preprocessing y split temporal para el pipeline provincial.
Todo ajuste de scalers e imputadores se hace exclusivamente sobre train.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ─────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────

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

TRAIN_YEARS = [2019, 2021, 2022, 2023]
VAL_YEARS   = [2024]
TEST_YEARS  = [2025, 2026]

CALENDAR_BINARY  = ["es_feriado"]
CALENDAR_ORDINAL = ["mes", "dia_semana"]


# ─────────────────────────────────────────────
# 1. Carga de datos
# ─────────────────────────────────────────────

def load_x(processed_dir: Path) -> pd.DataFrame:
    """
    Load the daily demand profiles matrix.
    Returns a DataFrame with columns: fecha, {province}_{hour:02d}, ...
    Shape: (n_days, 1 + 528)
    """
    path = processed_dir / "x_perfiles_diarios_provincias.csv"
    df = pd.read_csv(path, parse_dates=["fecha"])
    return df


def load_c(processed_dir: Path) -> pd.DataFrame:
    """
    Load the full condition matrix (climate + socioeconomic + calendar).
    Returns a DataFrame with columns: fecha, mes, dia_semana, es_feriado,
    climate features per province, socioeconomic features per province.
    """
    path = processed_dir / "c_condiciones_diarias.csv"
    df = pd.read_csv(path, parse_dates=["fecha"])
    return df


# ─────────────────────────────────────────────
# 2. Split temporal
# ─────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    date_col: str = "fecha",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into train, validation and test sets by year.

    Train:      2019 + 2021 + 2022 + 2023
    Validation: 2024
    Test:       2025 + Jan-Apr 2026

    Year 2020 is excluded from all splits (pandemic anomaly).
    """
    years = df[date_col].dt.year
    train = df[years.isin(TRAIN_YEARS)].reset_index(drop=True).copy()
    val   = df[years.isin(VAL_YEARS)].reset_index(drop=True).copy()
    test  = df[years.isin(TEST_YEARS)].reset_index(drop=True).copy()
    return train, val, test


def verify_split(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    date_col: str = "fecha",
) -> dict:
    """
    Verify the temporal split has no date overlap between sets.
    Returns a summary dict with sizes and date ranges.
    """
    train_dates = set(train[date_col])
    val_dates   = set(val[date_col])
    test_dates  = set(test[date_col])

    return {
        "train_size":         len(train),
        "val_size":           len(val),
        "test_size":          len(test),
        "train_years":        sorted(train[date_col].dt.year.unique().tolist()),
        "val_years":          sorted(val[date_col].dt.year.unique().tolist()),
        "test_years":         sorted(test[date_col].dt.year.unique().tolist()),
        "train_range":        (train[date_col].min(), train[date_col].max()),
        "val_range":          (val[date_col].min(),   val[date_col].max()),
        "test_range":         (test[date_col].min(),  test[date_col].max()),
        "overlap_train_val":  len(train_dates & val_dates),
        "overlap_train_test": len(train_dates & test_dates),
        "overlap_val_test":   len(val_dates   & test_dates),
        "leakage_detected":   bool(
            (train_dates & val_dates) or
            (train_dates & test_dates) or
            (val_dates   & test_dates)
        ),
    }


# ─────────────────────────────────────────────
# 3. Manejo de dias anomalos
# ─────────────────────────────────────────────

def detect_anomalous_days(x: pd.DataFrame) -> pd.DataFrame:
    """
    Identify days with null or simultaneous zero values across provinces.
    Returns a DataFrame with columns: fecha, reason.
    """
    demand_cols = [c for c in x.columns if c != "fecha"]
    records = []

    null_mask = x[demand_cols].isnull().any(axis=1)
    for fecha in x.loc[null_mask, "fecha"]:
        records.append({"fecha": fecha, "reason": "null values — possible outage"})

    zero_mask = (x[demand_cols] == 0).sum(axis=1) >= 3
    null_fechas = set(x.loc[null_mask, "fecha"])
    for fecha in x.loc[zero_mask, "fecha"]:
        if fecha not in null_fechas:
            records.append({"fecha": fecha, "reason": "simultaneous zeros — possible outage"})

    if not records:
        return pd.DataFrame(columns=["fecha", "reason"])
    return pd.DataFrame(records)


def remove_anomalous_days(
    x: pd.DataFrame,
    c: pd.DataFrame,
    extra_dates: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Remove anomalous days from x and c and return them separately
    for use as anomaly detection benchmark.

    Days are removed because they represent real power outages.
    Imputation would falsify the data.

    extra_dates: additional dates to remove beyond auto-detected ones.
    Returns: (x_clean, c_clean, anomaly_df)
    """
    x = x.reset_index(drop=True).copy()
    c = c.reset_index(drop=True).copy()

    anomaly_df = detect_anomalous_days(x)

    if extra_dates:
        existing = set(anomaly_df["fecha"])
        for d in extra_dates:
            fecha = pd.Timestamp(d)
            if fecha not in existing:
                anomaly_df = pd.concat([
                    anomaly_df,
                    pd.DataFrame([{"fecha": fecha, "reason": "manually specified"}])
                ], ignore_index=True)

    anomalous_dates = set(anomaly_df["fecha"])

    x_clean = x[~x["fecha"].isin(anomalous_dates)].reset_index(drop=True)
    c_clean = c[~c["fecha"].isin(anomalous_dates)].reset_index(drop=True)
    
    return x_clean, c_clean, anomaly_df


# ─────────────────────────────────────────────
# 4. Normalizacion de x
# ─────────────────────────────────────────────

def fit_x_scaler(
    x_train: pd.DataFrame,
    method: Literal["standard", "minmax"] = "standard",
) -> StandardScaler | MinMaxScaler:
    """
    Fit a scaler on training demand profiles.
    Fitted exclusively on train — never on validation or test.
    """
    demand_cols = [c for c in x_train.columns if c != "fecha"]
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    scaler.fit(x_train[demand_cols].values)
    return scaler


def transform_x(
    x: pd.DataFrame,
    scaler: StandardScaler | MinMaxScaler,
) -> np.ndarray:
    """
    Apply a pre-fitted scaler to a demand profile DataFrame.
    Returns a numpy array of shape (n_days, 528).
    The scaler must have been fitted on train only.
    """
    demand_cols = [c for c in x.columns if c != "fecha"]
    return scaler.transform(x[demand_cols].values)


def compute_daily_shares(x: pd.DataFrame) -> np.ndarray:
    """
    Normalize each daily profile by dividing each province-hour by the
    total national demand in that same hour.
    Returns a numpy array of shape (n_days, 528).
    No fitting needed — shares are scale-invariant.
    """
    demand_cols = [c for c in x.columns if c != "fecha"]
    values = x[demand_cols].values.reshape(-1, N_PROVINCES, N_HOURS)
    national_total = values.sum(axis=1, keepdims=True)
    shares = values / np.where(national_total == 0, 1, national_total)
    return shares.reshape(-1, X_DIM)


# ─────────────────────────────────────────────
# 5. Normalizacion de c
# ─────────────────────────────────────────────

def get_c_column_groups(c: pd.DataFrame) -> dict[str, list[str]]:
    """
    Identify which columns of c belong to each group:
    calendar_binary, calendar_ordinal, climate, socioeconomic.
    """
    all_cols = [col for col in c.columns if col != "fecha"]

    return {
        "calendar_binary":  [col for col in all_cols if col in CALENDAR_BINARY],
        "calendar_ordinal": [col for col in all_cols if col in CALENDAR_ORDINAL],
        "climate": [
            col for col in all_cols
            if any(col.startswith(p) for p in [
                "temp_", "sensacion_", "humedad_", "viento_", "grado_"
            ])
        ],
        "socioeconomic": [
            col for col in all_cols
            if any(col.startswith(p) for p in [
                "poblacion_", "pbi_", "participacion_industrial_", "superficie_"
            ])
        ],
    }


def fit_c_scaler(
    c_train: pd.DataFrame,
) -> tuple[StandardScaler, dict[str, list[str]]]:
    """
    Fit a StandardScaler on the continuous columns of c.
    Calendar binary columns are left as-is.
    Fitted exclusively on train.
    """
    groups = get_c_column_groups(c_train)
    cols_to_scale = groups["calendar_ordinal"] + groups["climate"] + groups["socioeconomic"]
    scaler = StandardScaler()
    scaler.fit(c_train[cols_to_scale].values)
    return scaler, groups


def transform_c(
    c: pd.DataFrame,
    scaler: StandardScaler,
    groups: dict[str, list[str]],
) -> np.ndarray:
    """
    Apply the pre-fitted scaler to c and return a numpy array.
    Binary calendar columns are concatenated without scaling.
    Column order: binary | scaled(ordinal + climate + socioeconomic).
    The scaler must have been fitted on train only.
    """
    cols_to_scale = groups["calendar_ordinal"] + groups["climate"] + groups["socioeconomic"]
    binary_vals = c[groups["calendar_binary"]].values
    scaled_vals = scaler.transform(c[cols_to_scale].values)
    return np.concatenate([binary_vals, scaled_vals], axis=1)


# ─────────────────────────────────────────────
# 6. Pipeline completo
# ─────────────────────────────────────────────

def run_preprocessing_pipeline(
    processed_dir: Path,
    x_scaler_method: Literal["standard", "minmax"] = "standard",
    save_dir: Path | None = None,
) -> dict:
    """
    Run the full preprocessing pipeline in the correct order to avoid leakage:

    1. Load x and c.
    2. Split by year into train / val / test.
    3. Verify no date overlap.
    4. Remove anomalous days (power outages).
    5. Fit x scaler on train only, transform all splits.
    6. Compute daily shares (no fitting needed).
    7. Fit c scaler on train only, transform all splits.
    8. Optionally save scalers and arrays to disk.
    """
    # 1. Load
    x_df = load_x(processed_dir)
    c_df = load_c(processed_dir)

    common_dates = set(x_df["fecha"]) & set(c_df["fecha"])
    x_df = x_df[x_df["fecha"].isin(common_dates)].sort_values("fecha").reset_index(drop=True)
    c_df = c_df[c_df["fecha"].isin(common_dates)].sort_values("fecha").reset_index(drop=True)
    print(f"Datos cargados: {len(x_df)} dias")

    # 2. Split
    x_train, x_val, x_test = temporal_split(x_df)
    c_train, c_val, c_test = temporal_split(c_df)

    # 3. Verify
    summary = verify_split(x_train, x_val, x_test)
    if summary["leakage_detected"]:
        raise ValueError("Leakage detectado en el split temporal.")
    print(f"Split: train={summary['train_size']}, val={summary['val_size']}, test={summary['test_size']}")

    # 4. Remove anomalous days
    x_train, c_train, anomaly_train = remove_anomalous_days(x_train, c_train)
    x_val,   c_val,   anomaly_val   = remove_anomalous_days(x_val,   c_val)
    x_test,  c_test,  anomaly_test  = remove_anomalous_days(x_test,  c_test)
    anomaly_log = pd.concat([anomaly_train, anomaly_val, anomaly_test], ignore_index=True)
    print(f"Dias anomalos eliminados: {len(anomaly_log)}")

    # 5. Fit x scaler on train, transform all
    x_scaler = fit_x_scaler(x_train, method=x_scaler_method)
    X_train = transform_x(x_train, x_scaler)
    X_val   = transform_x(x_val,   x_scaler)
    X_test  = transform_x(x_test,  x_scaler)

    # 6. Shares
    S_train = compute_daily_shares(x_train)
    S_val   = compute_daily_shares(x_val)
    S_test  = compute_daily_shares(x_test)

    # 7. Fit c scaler on train, transform all
    c_scaler, c_groups = fit_c_scaler(c_train)
    C_train = transform_c(c_train, c_scaler, c_groups)
    C_val   = transform_c(c_val,   c_scaler, c_groups)
    C_test  = transform_c(c_test,  c_scaler, c_groups)

    print(f"Shapes: X_train={X_train.shape}, C_train={C_train.shape}")

    result = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "C_train": C_train, "C_val": C_val, "C_test": C_test,
        "S_train": S_train, "S_val": S_val, "S_test": S_test,
        "dates_train": x_train["fecha"].values,
        "dates_val":   x_val["fecha"].values,
        "dates_test":  x_test["fecha"].values,
        "x_scaler": x_scaler,
        "c_scaler": c_scaler,
        "c_groups": c_groups,
        "x_dim":         X_train.shape[1],
        "c_dim":         C_train.shape[1],
        "split_summary": summary,
        "anomaly_log":   anomaly_log,
    }

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name in ["X_train", "X_val", "X_test", "C_train", "C_val", "C_test",
                     "S_train", "S_val", "S_test",
                     "dates_train", "dates_val", "dates_test"]:
            np.save(save_dir / f"{name}.npy", result[name])
        with open(save_dir / "x_scaler.pkl", "wb") as f:
            pickle.dump(x_scaler, f)
        with open(save_dir / "c_scaler.pkl", "wb") as f:
            pickle.dump(c_scaler, f)
        with open(save_dir / "c_groups.pkl", "wb") as f:
            pickle.dump(c_groups, f)
        anomaly_log.to_csv(save_dir / "anomaly_log.csv", index=False)
        print(f"Archivos guardados en {save_dir}/")

    return result
