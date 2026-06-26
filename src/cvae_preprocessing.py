from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

DEMAND_COLUMNS = [f"demanda_{h:02d}" for h in range(1, 25)]

OUTLIER_DATES = ["2023-03-01"]

DAY_TYPE_CATEGORIES = ["habil", "sabado", "domingo", "feriado"]

CYCLICAL_CONDITION_COLUMNS = {"mes": 12, "dia_semana": 7}
BINARY_CONDITION_COLUMNS = ["es_feriado"]
CONDITION_SKIP_COLUMNS = {"fecha", *CYCLICAL_CONDITION_COLUMNS, *BINARY_CONDITION_COLUMNS}


def remove_outlier_dates(
    x_profiles: pd.DataFrame,
    conditions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove known outlier dates from both X and C before splitting.

    The date 2023-03-01 corresponds to a massive blackout and is excluded
    from the baseline model for normal-demand profile generation.
    """
    outlier_dates = pd.to_datetime(OUTLIER_DATES)
    outlier_mask = x_profiles["fecha"].isin(outlier_dates)
    x_clean = x_profiles.loc[~outlier_mask].reset_index(drop=True)
    c_clean = conditions.loc[~outlier_mask].reset_index(drop=True)
    removed = int(outlier_mask.sum())
    if removed > 0:
        print(f"Fechas removidas: {removed} ({OUTLIER_DATES})")
    return x_clean, c_clean


def validate_daily_alignment(
    x_profiles: pd.DataFrame,
    conditions: pd.DataFrame,
) -> None:
    """Raise an error if demand profiles and conditions are not date-aligned."""
    x_dates = x_profiles["fecha"].reset_index(drop=True)
    c_dates = conditions["fecha"].reset_index(drop=True)
    if not x_dates.equals(c_dates):
        raise ValueError("X y C no tienen las mismas fechas en el mismo orden.")


def build_tipo_dia(conditions: pd.DataFrame) -> pd.Series:
    """
    Derive tipo_dia from dia_semana (0=Monday) and es_feriado.

    Categories: habil, sabado, domingo, feriado.
    Feriado takes priority over day of week.
    """
    def classify(row: pd.Series) -> str:
        if row["es_feriado"] == 1:
            return "feriado"
        if row["dia_semana"] == 5:
            return "sabado"
        if row["dia_semana"] == 6:
            return "domingo"
        return "habil"

    return conditions.apply(classify, axis=1)


def add_tipo_dia(conditions: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of conditions with the derived tipo_dia column."""
    enriched = conditions.copy()
    enriched["tipo_dia"] = build_tipo_dia(enriched)
    return enriched


def get_full_condition_numeric_columns(conditions: pd.DataFrame) -> list[str]:
    """
    Return numeric C-full columns to standardize.

    C-full includes all numeric condition columns except date, cyclical calendar
    columns and binary indicators already encoded separately.
    """
    return [
        column
        for column in conditions.columns
        if column not in CONDITION_SKIP_COLUMNS
        and pd.api.types.is_numeric_dtype(conditions[column])
    ]


def fit_condition_stats(conditions_train: pd.DataFrame) -> dict:
    """Fit standardization stats for C-full numeric columns using train only."""
    numeric_columns = get_full_condition_numeric_columns(conditions_train)
    numeric_stats = {}
    for column in numeric_columns:
        values = conditions_train[column].astype(np.float32).to_numpy()
        numeric_stats[column] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return {
        "encoding": "c_full",
        "day_type_categories": DAY_TYPE_CATEGORIES,
        "cyclical_columns": CYCLICAL_CONDITION_COLUMNS,
        "binary_columns": BINARY_CONDITION_COLUMNS,
        "numeric_columns": numeric_columns,
        "numeric_stats": numeric_stats,
    }


def build_condition_vector(
    conditions: pd.DataFrame,
    condition_stats: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Build the C-full conditioning matrix.

    Encoding:
    - tipo_dia: one-hot, 4 columns (habil, sabado, domingo, feriado)
    - mes: cyclical sin/cos, 2 columns
    - dia_semana: cyclical sin/cos, 2 columns
    - es_feriado: binary, 1 column
    - all remaining numeric condition columns standardized with train stats

    If condition_stats is None, stats are fitted on the provided data. For a
    leakage-free experiment, pass stats fitted on the training split.
    """
    if condition_stats is None:
        condition_stats = fit_condition_stats(conditions)

    parts = []
    feature_names = []

    tipo_dia = build_tipo_dia(conditions)
    tipo_dia_dummies = pd.get_dummies(tipo_dia, prefix="tipo").reindex(
        columns=[f"tipo_{category}" for category in DAY_TYPE_CATEGORIES],
        fill_value=0,
    )
    parts.append(tipo_dia_dummies.to_numpy(dtype=np.float32))
    feature_names.extend(tipo_dia_dummies.columns.tolist())

    for column, period in condition_stats["cyclical_columns"].items():
        values = conditions[column].to_numpy(dtype=np.float32)
        parts.append(np.sin(2 * np.pi * values / period).reshape(-1, 1).astype(np.float32))
        parts.append(np.cos(2 * np.pi * values / period).reshape(-1, 1).astype(np.float32))
        feature_names.extend([f"{column}_sin", f"{column}_cos"])

    for column in condition_stats["binary_columns"]:
        parts.append(conditions[[column]].to_numpy(dtype=np.float32))
        feature_names.append(column)

    for column in condition_stats["numeric_columns"]:
        values = conditions[column].to_numpy(dtype=np.float32)
        stats = condition_stats["numeric_stats"][column]
        scaled = (values - stats["mean"]) / (stats["std"] + 1e-8)
        parts.append(scaled.reshape(-1, 1).astype(np.float32))
        feature_names.append(column)

    c_matrix = np.column_stack(parts).astype(np.float32)
    metadata = {**condition_stats, "feature_names": feature_names}
    return c_matrix, metadata


def fit_profile_scaler(x_profiles_train: pd.DataFrame) -> dict:
    """
    Fit global profile standardization stats using train only.

    A single mean/std across all train days and hours preserves relative
    differences between hours.
    """
    X_train = x_profiles_train[DEMAND_COLUMNS].to_numpy(dtype=np.float32)
    return {
        "mean": float(X_train.mean()),
        "std": float(X_train.std()),
    }


def transform_profile_matrix(
    x_profiles: pd.DataFrame,
    scaler_stats: dict,
) -> np.ndarray:
    """Transform daily profiles to standardized X values."""
    X = x_profiles[DEMAND_COLUMNS].to_numpy(dtype=np.float32)
    return ((X - scaler_stats["mean"]) / (scaler_stats["std"] + 1e-8)).astype(np.float32)


def build_profile_matrix(x_profiles: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """
    Legacy helper: fit and transform profiles on the provided data.

    Prefer fit_profile_scaler + transform_profile_matrix after the temporal
    split to avoid leakage.
    """
    scaler_stats = fit_profile_scaler(x_profiles)
    X_scaled = transform_profile_matrix(x_profiles, scaler_stats)
    print(
        f"X shape: {X_scaled.shape} - "
        f"mean={scaler_stats['mean']:.1f} std={scaler_stats['std']:.1f} MWh"
    )
    return X_scaled, scaler_stats


def inverse_transform_profiles(X_scaled: np.ndarray, scaler_stats: dict) -> np.ndarray:
    """Convert standardized profiles back to MWh values."""
    return X_scaled * scaler_stats["std"] + scaler_stats["mean"]


def temporal_train_val_test_split(
    x_profiles: pd.DataFrame,
    conditions: pd.DataFrame,
    train_end_year: int = 2023,
    val_year: int = 2024,
    test_start_year: int = 2025,
) -> dict[str, dict[str, pd.DataFrame | pd.Series]]:
    """
    Split raw aligned dataframes into temporal train/val/test partitions.

    Default split:
    - train: years <= 2023
    - val: year == 2024
    - test: years >= 2025
    """
    validate_daily_alignment(x_profiles, conditions)
    years = conditions["fecha"].dt.year
    masks = {
        "train": years <= train_end_year,
        "val": years == val_year,
        "test": years >= test_start_year,
    }
    splits = {}
    for split_name, mask in masks.items():
        split_x = x_profiles.loc[mask].reset_index(drop=True)
        split_c = conditions.loc[mask].reset_index(drop=True)
        splits[split_name] = {
            "x_profiles": split_x,
            "conditions": split_c,
            "dates": split_x["fecha"].reset_index(drop=True),
        }
        print(
            f"{split_name}: {len(split_x)} dias "
            f"({split_x['fecha'].min().date()} a {split_x['fecha'].max().date()})"
        )
    return splits


def prepare_cvae_datasets(
    x_profiles: pd.DataFrame,
    conditions: pd.DataFrame,
    train_end_year: int = 2023,
    val_year: int = 2024,
    test_start_year: int = 2025,
    remove_outliers: bool = True,
) -> tuple[dict[str, dict], dict]:
    """
    Prepare leakage-free CVAE arrays and metadata for the final baseline.

    The profile scaler and C-full numeric scalers are fitted on train only and
    reused unchanged for val/test.
    """
    validate_daily_alignment(x_profiles, conditions)
    if remove_outliers:
        x_profiles, conditions = remove_outlier_dates(x_profiles, conditions)

    raw_splits = temporal_train_val_test_split(
        x_profiles=x_profiles,
        conditions=conditions,
        train_end_year=train_end_year,
        val_year=val_year,
        test_start_year=test_start_year,
    )

    x_scaler = fit_profile_scaler(raw_splits["train"]["x_profiles"])
    condition_stats = fit_condition_stats(raw_splits["train"]["conditions"])

    prepared = {}
    condition_metadata = None
    for split_name, split_data in raw_splits.items():
        X = transform_profile_matrix(split_data["x_profiles"], x_scaler)
        C, condition_metadata = build_condition_vector(
            split_data["conditions"],
            condition_stats=condition_stats,
        )
        prepared[split_name] = {
            **split_data,
            "conditions": add_tipo_dia(split_data["conditions"]),
            "X": X,
            "C": C,
        }
        print(f"{split_name}: X {X.shape} | C {C.shape}")

    preprocessing = {
        "x_scaler": x_scaler,
        "condition_metadata": condition_metadata,
        "split": {
            "train_end_year": train_end_year,
            "val_year": val_year,
            "test_start_year": test_start_year,
            "removed_outlier_dates": OUTLIER_DATES if remove_outliers else [],
        },
    }
    return prepared, preprocessing


def temporal_train_test_split(
    X: np.ndarray,
    C: np.ndarray,
    dates: pd.Series,
    test_year: int = 2025,
) -> tuple:
    """
    Legacy split helper kept for older notebooks.

    New experiments should use prepare_cvae_datasets so scalers are fitted on
    train only and a validation set is available.
    """
    test_mask = dates.dt.year >= test_year
    train_mask = ~test_mask

    X_train, X_test = X[train_mask], X[test_mask]
    C_train, C_test = C[train_mask], C[test_mask]
    dates_train = dates[train_mask].reset_index(drop=True)
    dates_test = dates[test_mask].reset_index(drop=True)

    print(f"Train: {X_train.shape[0]} dias (hasta {test_year - 1})")
    print(f"Test:  {X_test.shape[0]} dias ({test_year} en adelante)")

    return X_train, X_test, C_train, C_test, dates_train, dates_test


def build_dataloaders(
    X_train: np.ndarray,
    C_train: np.ndarray,
    X_val: np.ndarray,
    C_val: np.ndarray,
    X_test: np.ndarray | None = None,
    C_test: np.ndarray | None = None,
    batch_size: int = 64,
) -> tuple[DataLoader, ...]:
    """Build PyTorch DataLoaders for train/val and optional test sets."""
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(C_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(C_val)),
        batch_size=batch_size,
        shuffle=False,
    )
    if X_test is None or C_test is None:
        return train_loader, val_loader

    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(C_test)),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader

