from pathlib import Path

import pandas as pd


HEAT_THRESHOLD = 22.0
COLD_THRESHOLD = 15.0


def load_calendar(path: Path) -> pd.DataFrame:
    """Load the daily calendar variables used as base conditions."""
    calendar = pd.read_csv(path, parse_dates=["fecha"])
    expected_columns = ["fecha", "mes", "dia_semana", "es_feriado"]
    missing_columns = [col for col in expected_columns if col not in calendar.columns]
    if missing_columns:
        raise ValueError(f"Missing calendar columns: {missing_columns}")
    return calendar[expected_columns].copy()


def load_hourly_weather(path: Path) -> pd.DataFrame:
    """Load hourly effective weather variables."""
    weather = pd.read_csv(path, parse_dates=["fecha"])
    expected_columns = [
        "fecha",
        "hora",
        "temp_ef",
        "humedad_ef",
        "viento_ef",
        "sensacion_ef",
    ]
    missing_columns = [col for col in expected_columns if col not in weather.columns]
    if missing_columns:
        raise ValueError(f"Missing weather columns: {missing_columns}")
    return weather[expected_columns].copy()


def build_daily_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly weather into one row per day."""
    daily_weather = (
        weather.groupby("fecha")
        .agg(
            temp_media=("temp_ef", "mean"),
            temp_min=("temp_ef", "min"),
            temp_max=("temp_ef", "max"),
            humedad_media=("humedad_ef", "mean"),
            viento_media=("viento_ef", "mean"),
            sensacion_media=("sensacion_ef", "mean"),
            sensacion_min=("sensacion_ef", "min"),
            sensacion_max=("sensacion_ef", "max"),
        )
        .reset_index()
    )
    daily_weather["temp_amplitud"] = daily_weather["temp_max"] - daily_weather["temp_min"]
    daily_weather["sensacion_amplitud"] = (
        daily_weather["sensacion_max"] - daily_weather["sensacion_min"]
    )
    daily_weather["grado_calor"] = (daily_weather["temp_media"] - HEAT_THRESHOLD).clip(
        lower=0
    )
    daily_weather["grado_frio"] = (COLD_THRESHOLD - daily_weather["temp_media"]).clip(
        lower=0
    )
    return daily_weather


def validate_daily_alignment(conditions: pd.DataFrame, x_profiles_path: Path) -> None:
    """Validate that condition dates match demand profile dates exactly."""
    x_profiles = pd.read_csv(x_profiles_path, usecols=["fecha"], parse_dates=["fecha"])
    condition_dates = conditions["fecha"].reset_index(drop=True)
    profile_dates = x_profiles["fecha"].reset_index(drop=True)
    if len(condition_dates) != len(profile_dates):
        raise ValueError(
            f"Different row counts: C={len(condition_dates)}, X={len(profile_dates)}"
        )
    if not condition_dates.equals(profile_dates):
        raise ValueError("Condition dates are not aligned with X profile dates")
    if conditions.isna().any().any():
        null_counts = conditions.isna().sum()
        raise ValueError(f"Found null values:\n{null_counts[null_counts > 0]}")


def build_conditions_dataset(
    calendar_path: Path,
    weather_path: Path,
    x_profiles_path: Path,
) -> pd.DataFrame:
    """Build the final daily condition matrix for conditional generative models."""
    calendar = load_calendar(calendar_path)
    weather = load_hourly_weather(weather_path)
    daily_weather = build_daily_weather(weather)
    conditions = calendar.merge(daily_weather, on="fecha", how="left")
    final_columns = [
        "fecha",
        "mes",
        "dia_semana",
        "es_feriado",
        "temp_media",
        "temp_min",
        "temp_max",
        "temp_amplitud",
        "humedad_media",
        "viento_media",
        "sensacion_media",
        "sensacion_min",
        "sensacion_max",
        "sensacion_amplitud",
        "grado_calor",
        "grado_frio",
    ]
    conditions = conditions[final_columns].sort_values("fecha").reset_index(drop=True)
    validate_daily_alignment(conditions, x_profiles_path)
    return conditions


def main() -> None:
    """Create c_condiciones_diarias.csv from calendar and hourly weather data."""
    conditions = build_conditions_dataset(
        calendar_path=Path("c_calendario.csv"),
        weather_path=Path("clima_efectivo_horario.csv"),
        x_profiles_path=Path("x_perfiles_diarios.csv"),
    )
    conditions.to_csv("c_condiciones_diarias.csv", index=False, float_format="%.6f")
    print(f"Saved c_condiciones_diarias.csv with shape {conditions.shape}")
    print(f"Date range: {conditions['fecha'].min().date()} -> {conditions['fecha'].max().date()}")
    print(conditions.head().to_string(index=False))


if __name__ == "__main__":
    main()
