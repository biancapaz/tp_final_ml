from pathlib import Path

import pandas as pd


PROVINCE_ORDER = [
    "buenos_aires", "catamarca", "chaco", "chubut", "cordoba",
    "corrientes", "entre_rios", "formosa", "jujuy", "la_pampa",
    "la_rioja", "mendoza", "misiones", "neuquen", "rio_negro",
    "salta", "san_juan", "san_luis", "santa_cruz", "santa_fe",
    "sgo_del_estero", "tucuman",
]
N_HOURS = 24

RAW_SOURCES = {
    "2019_2022": "dataset_demanda_horaria_provincia_2019_2022.csv",
    "2023_2026": "dataset_demanda_horaria_provincia_2023_2026.csv",
}


def load_provincial_datasets(project_root: Path) -> dict[str, pd.DataFrame]:
    """Load the processed provincial datasets used for the quality checks."""
    processed_path = project_root / "provincias" / "data" / "processed"
    return {
        "x_perfiles_diarios": pd.read_csv(
            processed_path / "x_perfiles_diarios_provincias.csv", parse_dates=["fecha"]
        ),
        "dataset_base_horario": pd.read_csv(
            processed_path / "dataset_base_horario.csv", parse_dates=["fecha"]
        ),
        "c_calendario": pd.read_csv(
            processed_path / "c_calendario.csv", parse_dates=["fecha"]
        ),
        "clima_diario": pd.read_csv(
            processed_path / "clima_diario_por_provincia.csv", parse_dates=["fecha"]
        ),
    }


def load_national_hourly_demand(project_root: Path) -> pd.DataFrame:
    """Load the national hourly demand series used as an independent reference total."""
    return pd.read_csv(project_root / "dataset_base_horario.csv", parse_dates=["fecha"])


def summarize_date_coverage(x_profiles: pd.DataFrame) -> dict[str, object]:
    """Summarize the date range and missing days of the provincial daily dataset."""
    dates = x_profiles["fecha"].drop_duplicates().sort_values()
    expected_dates = pd.date_range(dates.min(), dates.max(), freq="D")
    expected_dates = expected_dates[expected_dates.year != 2020]
    missing_dates = expected_dates.difference(dates)
    return {
        "start_date": dates.min().date(),
        "end_date": dates.max().date(),
        "observed_days": len(dates),
        "expected_days": len(expected_dates),
        "missing_days": len(missing_dates),
        "years_present": sorted(dates.dt.year.unique().tolist()),
    }


def count_hours_per_day(hourly_demand: pd.DataFrame) -> pd.Series:
    """Count how many hourly records are available for each date."""
    return hourly_demand.groupby("fecha")["hora"].count()


def count_duplicate_hourly_records(hourly_demand: pd.DataFrame) -> int:
    """Count duplicated date-hour records in the hourly dataset."""
    return int(hourly_demand.duplicated(["fecha", "hora"]).sum())


def summarize_missing_by_province(x_profiles: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing demand values for each province across all daily profiles."""
    rows = []
    for province in PROVINCE_ORDER:
        province_columns = [f"{province}_{hour:02d}" for hour in range(1, N_HOURS + 1)]
        missing_mask = x_profiles[province_columns].isna().any(axis=1)
        rows.append(
            {
                "provincia": province,
                "valores_faltantes": int(x_profiles[province_columns].isna().sum().sum()),
                "dias_afectados": int(missing_mask.sum()),
                "primera_fecha_afectada": x_profiles.loc[missing_mask, "fecha"].min()
                if missing_mask.any()
                else pd.NaT,
                "ultima_fecha_afectada": x_profiles.loc[missing_mask, "fecha"].max()
                if missing_mask.any()
                else pd.NaT,
            }
        )
    summary = pd.DataFrame(rows)
    return summary[summary["valores_faltantes"] > 0].reset_index(drop=True)


def summarize_zero_values_by_province(hourly_demand: pd.DataFrame) -> pd.DataFrame:
    """Summarize zero and negative hourly demand values for each province."""
    rows = []
    for province in PROVINCE_ORDER:
        values = hourly_demand[province]
        zero_mask = values == 0
        rows.append(
            {
                "provincia": province,
                "valores_cero": int(zero_mask.sum()),
                "valores_negativos": int((values < 0).sum()),
                "primera_fecha_cero": hourly_demand.loc[zero_mask, "fecha"].min()
                if zero_mask.any()
                else pd.NaT,
                "ultima_fecha_cero": hourly_demand.loc[zero_mask, "fecha"].max()
                if zero_mask.any()
                else pd.NaT,
            }
        )
    summary = pd.DataFrame(rows)
    return summary[(summary["valores_cero"] > 0) | (summary["valores_negativos"] > 0)].reset_index(
        drop=True
    )


def compare_provincial_sum_to_national_series(
    hourly_demand: pd.DataFrame,
    national_hourly_demand: pd.DataFrame,
) -> pd.DataFrame:
    """Compare the sum of the 22 provinces against the national hourly demand series."""
    merged = hourly_demand.merge(
        national_hourly_demand[["fecha", "hora", "demanda_mwh"]],
        on=["fecha", "hora"],
        how="inner",
    )
    merged["suma_provincias"] = merged[PROVINCE_ORDER].sum(axis=1)
    merged["diferencia_absoluta"] = (merged["suma_provincias"] - merged["demanda_mwh"]).abs()
    merged["diferencia_pct"] = merged["diferencia_absoluta"] / merged["demanda_mwh"] * 100
    return pd.DataFrame(
        [
            {
                "registros_comparados": len(merged),
                "diferencia_absoluta_media_mwh": merged["diferencia_absoluta"].mean(),
                "diferencia_absoluta_maxima_mwh": merged["diferencia_absoluta"].max(),
                "diferencia_pct_media": merged["diferencia_pct"].mean(),
                "diferencia_pct_maxima": merged["diferencia_pct"].max(),
                "registros_con_diferencia_mayor_a_1pct": int((merged["diferencia_pct"] > 1).sum()),
            }
        ]
    )


def parse_argentine_number(value: object) -> float:
    """Parse a number formatted with '.' as thousands separator and ',' as decimal."""
    if pd.isna(value):
        return float("nan")
    text = str(value).strip().replace(".", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def load_raw_provincial_source(path: Path) -> pd.DataFrame:
    """Load a raw CAMMESA provincial demand CSV keeping only province columns and TOTAL."""
    data = pd.read_csv(path, sep=";", skiprows=3, encoding="latin-1", dtype=str)
    data = data.dropna(axis=1, how="all")
    data.columns = [column.strip() for column in data.columns]
    normalized_names = {
        column: column.strip().lower().replace(".", "_").replace(" ", "_")
        for column in data.columns
    }
    province_columns = {
        column: normalized
        for column, normalized in normalized_names.items()
        if normalized in PROVINCE_ORDER
    }
    for column in list(province_columns) + ["TOTAL"]:
        data[column] = data[column].apply(parse_argentine_number)
    data = data.rename(columns=province_columns)
    return data[list(province_columns.values()) + ["TOTAL"]]


def summarize_raw_total_consistency(project_root: Path) -> pd.DataFrame:
    """Compare the provincial sum against the TOTAL column in each raw source file."""
    raw_path = project_root / "provincias" / "data" / "raw"
    rows = []
    for source_name, file_name in RAW_SOURCES.items():
        data = load_raw_provincial_source(raw_path / file_name)
        province_columns = [column for column in PROVINCE_ORDER if column in data.columns]
        province_sum = data[province_columns].sum(axis=1)
        absolute_difference = (province_sum - data["TOTAL"]).abs()
        rows.append(
            {
                "fuente": source_name,
                "registros": len(data),
                "provincias_encontradas": len(province_columns),
                "diferencia_absoluta_media_mwh": absolute_difference.mean(),
                "diferencia_absoluta_maxima_mwh": absolute_difference.max(),
                "registros_con_diferencia_mayor_a_1_mwh": int((absolute_difference > 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_provincial_quality_summary(project_root: Path) -> dict[str, object]:
    """Run the main provincial dataset quality checks used in the EDA."""
    datasets = load_provincial_datasets(project_root)
    national_hourly = load_national_hourly_demand(project_root)
    x_profiles = datasets["x_perfiles_diarios"]
    hourly_demand = datasets["dataset_base_horario"]
    return {
        "date_coverage": summarize_date_coverage(x_profiles),
        "hours_per_day_counts": count_hours_per_day(hourly_demand).value_counts().sort_index(),
        "duplicate_hourly_records": count_duplicate_hourly_records(hourly_demand),
        "missing_by_province": summarize_missing_by_province(x_profiles),
        "zero_values_by_province": summarize_zero_values_by_province(hourly_demand),
        "raw_total_consistency": summarize_raw_total_consistency(project_root),
        "provincial_sum_vs_national_series": compare_provincial_sum_to_national_series(
            hourly_demand, national_hourly
        ),
    }
