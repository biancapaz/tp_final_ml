import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

DEMAND_COLUMNS = [f"demanda_{h:02d}" for h in range(1, 25)]

OUTLIER_DATES = ["2023-03-01"]

DAY_TYPE_CATEGORIES = ["habil", "sabado", "domingo", "feriado"]


def remove_outlier_dates(
    x_profiles: pd.DataFrame,
    conditions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove known outlier dates from both X and C before training.

    The date 2023-03-01 corresponds to a massive blackout affecting
    20 million people and does not represent normal system behavior.
    """
    outlier_mask = x_profiles["fecha"].isin(pd.to_datetime(OUTLIER_DATES))
    x_clean = x_profiles[~outlier_mask].reset_index(drop=True)
    c_clean = conditions[~outlier_mask].reset_index(drop=True)
    removed = outlier_mask.sum()
    if removed > 0:
        print(f"Fechas removidas: {removed} ({OUTLIER_DATES})")
    return x_clean, c_clean


def build_tipo_dia(conditions: pd.DataFrame) -> pd.Series:
    """
    Derive tipo_dia from dia_semana (0=Monday) and es_feriado.

    Categories: habil, sabado, domingo, feriado.
    Feriado takes priority over day of week.
    """
    def classify(row):
        if row["es_feriado"] == 1:
            return "feriado"
        elif row["dia_semana"] == 5:
            return "sabado"
        elif row["dia_semana"] == 6:
            return "domingo"
        else:
            return "habil"
    return conditions.apply(classify, axis=1)


def build_condition_vector(conditions: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """
    Build the conditioning matrix C from all available variables.

    Encoding:
    - tipo_dia (derived): one-hot, 4 columns (habil, sabado, domingo, feriado)
    - mes: cyclical sin/cos, 2 columns
    - dia_semana: cyclical sin/cos, 2 columns
    - es_feriado: binary, 1 column
    - all remaining numeric columns: standardized, 1 column each
      (temp_media, temp_min, temp_max, temp_amplitud, humedad_media,
       viento_media, sensacion_media, sensacion_min, sensacion_max,
       sensacion_amplitud, grado_calor, grado_frio)

    Total: 4 + 2 + 2 + 1 + 12 = 21 dimensions.
    """
    parts = []
    stats = {}

    # derive and one-hot encode tipo_dia
    tipo_dia = build_tipo_dia(conditions)
    tipo_dia_dummies = pd.get_dummies(tipo_dia, prefix="tipo").reindex(
        columns=[f"tipo_{t}" for t in DAY_TYPE_CATEGORIES], fill_value=0
    )
    parts.append(tipo_dia_dummies.values.astype(np.float32))

    # cyclical encoding for mes and dia_semana
    for col, period in [("mes", 12), ("dia_semana", 7)]:
        values = conditions[col].values
        parts.append(np.sin(2 * np.pi * values / period).reshape(-1, 1).astype(np.float32))
        parts.append(np.cos(2 * np.pi * values / period).reshape(-1, 1).astype(np.float32))

    # binary es_feriado as-is
    parts.append(conditions[["es_feriado"]].values.astype(np.float32))

    # standardize all remaining numeric columns
    skip = {"fecha", "mes", "dia_semana", "es_feriado"}
    numeric_cols = [
        c for c in conditions.columns
        if c not in skip and conditions[c].dtype != object
    ]
    for col in numeric_cols:
        values = conditions[col].values.astype(np.float32)
        mean, std = float(values.mean()), float(values.std())
        parts.append(((values - mean) / (std + 1e-8)).reshape(-1, 1))
        stats[col] = {"mean": mean, "std": std}

    c_matrix = np.column_stack(parts).astype(np.float32)
    print(f"C shape: {c_matrix.shape}  — {c_matrix.shape[1]} dimensiones")
    return c_matrix, stats


def build_profile_matrix(x_profiles: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """
    Build the X matrix of demand profiles and standardize it globally.

    A single mean and std across all hours and days is used so that
    relative differences between hours are preserved.

    Returns the scaled matrix and the stats needed to invert later.
    """
    X = x_profiles[DEMAND_COLUMNS].values.astype(np.float32)
    x_mean = float(X.mean())
    x_std = float(X.std())
    X_scaled = (X - x_mean) / x_std
    stats = {"mean": x_mean, "std": x_std}
    print(f"X shape: {X_scaled.shape}  — mean={x_mean:.1f} std={x_std:.1f} MWh")
    return X_scaled, stats


def inverse_transform_profiles(X_scaled: np.ndarray, scaler_stats: dict) -> np.ndarray:
    """Convert standardized profiles back to MWh values."""
    return X_scaled * scaler_stats["std"] + scaler_stats["mean"]


def temporal_train_test_split(
    X: np.ndarray,
    C: np.ndarray,
    dates: pd.Series,
    test_year: int = 2025,
) -> tuple:
    """
    Split X and C into train and test sets using a temporal cutoff.

    All days from test_year onward go to test. This avoids data leakage
    since the model could otherwise memorize recent patterns.
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
    X_test: np.ndarray,
    C_test: np.ndarray,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader]:
    """Build PyTorch DataLoaders for train and test sets."""
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(C_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(C_test)),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader