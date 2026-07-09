from pathlib import Path

import pandas as pd
import unicodedata


DEMAND_COLUMNS = [f"demanda_{hour:02d}" for hour in range(1, 25)]
RAW_DATA_DIR = Path("data/raw")
INTERIM_DATA_DIR = Path("data/interim")
PROCESSED_DATA_DIR = Path("data/processed")


def load_daily_datasets(
    x_path: Path = PROCESSED_DATA_DIR / "x_perfiles_diarios.csv",
    c_path: Path = INTERIM_DATA_DIR / "c_condiciones_diarias.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the daily demand profiles and condition datasets."""
    x_profiles = pd.read_csv(x_path, parse_dates=["fecha"])
    conditions = pd.read_csv(c_path, parse_dates=["fecha"])
    return x_profiles, conditions


def load_hourly_datasets(
    demand_path: Path = PROCESSED_DATA_DIR / "dataset_base_horario.csv",
    weather_path: Path = INTERIM_DATA_DIR / "clima_efectivo_horario.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the hourly demand and weather datasets."""
    hourly_demand = pd.read_csv(demand_path, parse_dates=["fecha"])
    hourly_weather = pd.read_csv(weather_path, parse_dates=["fecha"])
    return hourly_demand, hourly_weather


def normalize_column_name(column: object) -> str:
    """Normalize source spreadsheet column names to simple snake case."""
    if not isinstance(column, str):
        return str(column)
    normalized = unicodedata.normalize("NFKD", column)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.strip().lower()
    normalized = normalized.replace(" ", "_")
    normalized = normalized.replace(".", "")
    normalized = normalized.replace("-", "_")
    normalized = normalized.replace("%", "pct")
    normalized = normalized.replace("°", "")
    return normalized


def find_header_row(
    path: Path,
    required_columns: list[str],
    max_rows: int = 20,
) -> int:
    """Find the row that contains the expected source spreadsheet headers."""
    raw_data = pd.read_excel(path, header=None, nrows=max_rows)
    required_set = {normalize_column_name(column) for column in required_columns}
    for row_index, row in raw_data.iterrows():
        normalized_values = {
            normalize_column_name(value) for value in row.tolist() if pd.notna(value)
        }
        if required_set.issubset(normalized_values):
            return int(row_index)
    raise ValueError(f"Could not find header row in {path}")


def load_source_spreadsheet(path: Path, required_columns: list[str]) -> pd.DataFrame:
    """Load a source Excel file using its detected header row."""
    header_row = find_header_row(path, required_columns)
    data = pd.read_excel(path, header=header_row)
    data.columns = [normalize_column_name(column) for column in data.columns]
    return data


def summarize_source_total_consistency(base_path: Path = Path(".")) -> pd.DataFrame:
    """Compare regional or provincial sums against the source TOTAL column."""
    source_specs = [
        {
            "dataset": "dataset_2021_2023",
            "path": base_path / RAW_DATA_DIR / "dataset_2021_2023.xlsx",
            "required_columns": ["AÑO", "MES", "DÍA", "TIPO DÍA", "HORA"],
            "metadata_columns": {"ano", "mes", "dia", "tipo_dia", "hora"},
            "excluded_columns": {
                "total",
                "crono",
                "orden",
                "supera_200_hs",
                "crono1",
                "orden1",
                "supera_200_hs1",
            },
        },
        {
            "dataset": "dataset_2023_2026",
            "path": base_path / RAW_DATA_DIR / "dataset_2023_2026.xlsx",
            "required_columns": [
                "AÑO",
                "MES",
                "N° MES",
                "N° DIA",
                "TIPO DÍA",
                "DÍA",
                "FECHA",
                "HORA",
            ],
            "metadata_columns": {
                "ano",
                "mes",
                "n_mes",
                "n_dia",
                "tipo_dia",
                "dia",
                "fecha",
                "hora",
            },
            "excluded_columns": {"total"},
        },
    ]
    rows = []
    for spec in source_specs:
        data = load_source_spreadsheet(spec["path"], spec["required_columns"])
        demand_columns = [
            column
            for column in data.columns
            if column not in spec["metadata_columns"]
            and column not in spec["excluded_columns"]
            and not column.startswith("unnamed")
        ]
        source_sum = data[demand_columns].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        total = pd.to_numeric(data["total"], errors="coerce")
        absolute_difference = (source_sum - total).abs()
        rows.append(
            {
                "dataset": spec["dataset"],
                "rows": len(data),
                "demand_columns": len(demand_columns),
                "max_abs_difference": absolute_difference.max(),
                "rows_above_001_mwh": int((absolute_difference > 0.01).sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_source_transition(hourly_demand: pd.DataFrame) -> pd.DataFrame:
    """Summarize demand around the transition from 2022 to 2023."""
    daily_demand = (
        hourly_demand.groupby("fecha")
        .agg(
            energia_diaria=("demanda_mwh", "sum"),
            demanda_media=("demanda_mwh", "mean"),
            demanda_maxima=("demanda_mwh", "max"),
        )
        .reset_index()
    )
    periods = {
        "2022_last_30_days": ("2022-12-02", "2022-12-31"),
        "2023_first_30_days": ("2023-01-01", "2023-01-30"),
    }
    rows = []
    for name, (start_date, end_date) in periods.items():
        mask = daily_demand["fecha"].between(start_date, end_date)
        period_data = daily_demand.loc[mask]
        rows.append(
            {
                "period": name,
                "days": len(period_data),
                "energia_media_diaria": period_data["energia_diaria"].mean(),
                "demanda_media_horaria": period_data["demanda_media"].mean(),
                "demanda_maxima_media": period_data["demanda_maxima"].mean(),
            }
        )
    return pd.DataFrame(rows)


def summarize_date_coverage(data: pd.DataFrame) -> dict[str, object]:
    """Summarize date coverage for a dataset with a fecha column."""
    dates = data["fecha"].drop_duplicates().sort_values()
    expected_dates = pd.date_range(dates.min(), dates.max(), freq="D")
    missing_dates = expected_dates.difference(dates)
    return {
        "start_date": dates.min().date(),
        "end_date": dates.max().date(),
        "observed_days": len(dates),
        "expected_days": len(expected_dates),
        "missing_days": len(missing_dates),
    }


def count_hours_per_day(hourly_data: pd.DataFrame) -> pd.Series:
    """Count the number of hourly records available for each date."""
    return hourly_data.groupby("fecha")["hora"].count()


def count_duplicate_hourly_records(hourly_data: pd.DataFrame) -> int:
    """Count duplicated records at the date-hour level."""
    return int(hourly_data.duplicated(["fecha", "hora"]).sum())


def validate_daily_alignment(
    x_profiles: pd.DataFrame,
    conditions: pd.DataFrame,
) -> bool:
    """Check whether X and C have the same dates in the same order."""
    x_dates = x_profiles["fecha"].reset_index(drop=True)
    c_dates = conditions["fecha"].reset_index(drop=True)
    return bool(x_dates.equals(c_dates))


def summarize_missing_values(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return total missing values for each dataset."""
    rows = []
    for name, data in datasets.items():
        rows.append(
            {
                "dataset": name,
                "rows": len(data),
                "columns": len(data.columns),
                "missing_values": int(data.isna().sum().sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_value_ranges(
    x_profiles: pd.DataFrame,
    conditions: pd.DataFrame,
    hourly_demand: pd.DataFrame,
    hourly_weather: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize important numeric ranges for basic plausibility checks."""
    rows = [
        {
            "variable": "demanda_mwh",
            "min": hourly_demand["demanda_mwh"].min(),
            "max": hourly_demand["demanda_mwh"].max(),
        },
        {
            "variable": "X_demanda_min",
            "min": x_profiles[DEMAND_COLUMNS].min().min(),
            "max": x_profiles[DEMAND_COLUMNS].max().max(),
        },
        {
            "variable": "temp_ef",
            "min": hourly_weather["temp_ef"].min(),
            "max": hourly_weather["temp_ef"].max(),
        },
        {
            "variable": "humedad_ef",
            "min": hourly_weather["humedad_ef"].min(),
            "max": hourly_weather["humedad_ef"].max(),
        },
        {
            "variable": "viento_ef",
            "min": hourly_weather["viento_ef"].min(),
            "max": hourly_weather["viento_ef"].max(),
        },
        {
            "variable": "sensacion_ef",
            "min": hourly_weather["sensacion_ef"].min(),
            "max": hourly_weather["sensacion_ef"].max(),
        },
        {
            "variable": "grado_calor",
            "min": conditions["grado_calor"].min(),
            "max": conditions["grado_calor"].max(),
        },
        {
            "variable": "grado_frio",
            "min": conditions["grado_frio"].min(),
            "max": conditions["grado_frio"].max(),
        },
    ]
    return pd.DataFrame(rows)


def build_quality_summary(
    base_path: Path = Path("."),
    include_source_excel_checks: bool = False,
) -> dict[str, object]:
    """Run the main dataset quality checks used in the EDA."""
    x_profiles, conditions = load_daily_datasets(
        x_path=base_path / PROCESSED_DATA_DIR / "x_perfiles_diarios.csv",
        c_path=base_path / INTERIM_DATA_DIR / "c_condiciones_diarias.csv",
    )
    hourly_demand, hourly_weather = load_hourly_datasets(
        demand_path=base_path / PROCESSED_DATA_DIR / "dataset_base_horario.csv",
        weather_path=base_path / INTERIM_DATA_DIR / "clima_efectivo_horario.csv",
    )
    demand_hours_per_day = count_hours_per_day(hourly_demand)
    weather_hours_per_day = count_hours_per_day(hourly_weather)
    datasets = {
        "x_perfiles_diarios": x_profiles,
        "c_condiciones_diarias": conditions,
        "dataset_base_horario": hourly_demand,
        "clima_efectivo_horario": hourly_weather,
    }
    summary = {
        "x_profiles_shape": x_profiles.shape,
        "conditions_shape": conditions.shape,
        "hourly_demand_shape": hourly_demand.shape,
        "hourly_weather_shape": hourly_weather.shape,
        "date_coverage": summarize_date_coverage(x_profiles),
        "x_c_aligned": validate_daily_alignment(x_profiles, conditions),
        "demand_hour_counts": demand_hours_per_day.value_counts().sort_index(),
        "weather_hour_counts": weather_hours_per_day.value_counts().sort_index(),
        "demand_duplicates": count_duplicate_hourly_records(hourly_demand),
        "weather_duplicates": count_duplicate_hourly_records(hourly_weather),
        "missing_values": summarize_missing_values(datasets),
        "value_ranges": summarize_value_ranges(
            x_profiles,
            conditions,
            hourly_demand,
            hourly_weather,
        ),
        "source_transition": summarize_source_transition(hourly_demand),
        "leap_day_present": pd.Timestamp("2024-02-29")
        in set(x_profiles["fecha"]),
        "last_year_is_partial": x_profiles["fecha"].max().year == 2026
        and x_profiles["fecha"].max().month < 12,
    }
    if include_source_excel_checks:
        summary["source_total_consistency"] = summarize_source_total_consistency(
            base_path
        )
    return summary
