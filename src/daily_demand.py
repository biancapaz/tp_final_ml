from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes


DEMAND_COLUMNS = [f"demanda_{hour:02d}" for hour in range(1, 25)]
HOURS = list(range(1, 25))
DAY_TYPE_ORDER = ["habil", "sabado", "domingo", "feriado"]


def load_daily_model_data(base_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load daily demand profiles and daily condition variables."""
    x_profiles = pd.read_csv(
        base_path / "data" / "processed" / "x_perfiles_diarios.csv",
        parse_dates=["fecha"],
    )
    conditions = pd.read_csv(
        base_path / "data" / "interim" / "c_condiciones_diarias.csv",
        parse_dates=["fecha"],
    )
    return x_profiles, conditions


def assign_day_type(row: pd.Series) -> str:
    """Assign a readable calendar type to one daily row."""
    if row["es_feriado"] == 1:
        return "feriado"
    if row["dia_semana"] == 5:
        return "sabado"
    if row["dia_semana"] == 6:
        return "domingo"
    return "habil"


def build_daily_demand_features(
    x_profiles: pd.DataFrame,
    conditions: pd.DataFrame,
) -> pd.DataFrame:
    """Add daily demand summary features to the daily profile dataset."""
    calendar_columns = ["fecha", "mes", "dia_semana", "es_feriado"]
    daily_demand = x_profiles.merge(conditions[calendar_columns], on="fecha", how="left")
    daily_demand["energia_diaria_mwh"] = daily_demand[DEMAND_COLUMNS].sum(axis=1)
    daily_demand["demanda_media_mwh"] = daily_demand[DEMAND_COLUMNS].mean(axis=1)
    daily_demand["demanda_maxima_mwh"] = daily_demand[DEMAND_COLUMNS].max(axis=1)
    daily_demand["demanda_minima_mwh"] = daily_demand[DEMAND_COLUMNS].min(axis=1)
    daily_demand["hora_pico"] = (
        daily_demand[DEMAND_COLUMNS].idxmax(axis=1).str[-2:].astype(int)
    )
    daily_demand["hora_valle"] = (
        daily_demand[DEMAND_COLUMNS].idxmin(axis=1).str[-2:].astype(int)
    )
    daily_demand["amplitud_diaria_mwh"] = (
        daily_demand["demanda_maxima_mwh"] - daily_demand["demanda_minima_mwh"]
    )
    daily_demand["factor_carga"] = (
        daily_demand["demanda_media_mwh"] / daily_demand["demanda_maxima_mwh"]
    )
    daily_demand["tipo_dia"] = daily_demand.apply(assign_day_type, axis=1)
    return daily_demand


def summarize_daily_demand(daily_demand: pd.DataFrame) -> pd.DataFrame:
    """Summarize the main daily demand indicators."""
    summary_columns = [
        "energia_diaria_mwh",
        "demanda_media_mwh",
        "demanda_maxima_mwh",
        "demanda_minima_mwh",
        "hora_pico",
        "hora_valle",
        "amplitud_diaria_mwh",
        "factor_carga",
    ]
    return (
        daily_demand[summary_columns]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
        .round(2)
    )


def summarize_by_month(daily_demand: pd.DataFrame) -> pd.DataFrame:
    """Summarize daily energy and peak demand by calendar month."""
    return (
        daily_demand.groupby("mes")
        .agg(
            energia_media_gwh=("energia_diaria_mwh", lambda values: values.mean() / 1000),
            demanda_maxima_media_mwh=("demanda_maxima_mwh", "mean"),
            hora_pico_mediana=("hora_pico", "median"),
        )
        .round(2)
    )


def summarize_by_day_type(daily_demand: pd.DataFrame) -> pd.DataFrame:
    """Summarize daily demand indicators by day type."""
    summary = (
        daily_demand.groupby("tipo_dia")
        .agg(
            dias=("fecha", "count"),
            energia_media_gwh=("energia_diaria_mwh", lambda values: values.mean() / 1000),
            demanda_maxima_media_mwh=("demanda_maxima_mwh", "mean"),
            factor_carga_medio=("factor_carga", "mean"),
        )
        .reindex(DAY_TYPE_ORDER)
    )
    return summary.round(2)


def get_peak_hour_distribution(daily_demand: pd.DataFrame) -> pd.DataFrame:
    """Return the daily peak-hour distribution."""
    counts = daily_demand["hora_pico"].value_counts().sort_index()
    distribution = counts.rename_axis("hora_pico").reset_index(name="dias")
    distribution["porcentaje"] = distribution["dias"] / len(daily_demand) * 100
    return distribution.round({"porcentaje": 1})


def plot_average_daily_profile(daily_demand: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot the average hourly demand profile with an interquartile band."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    average_profile = daily_demand[DEMAND_COLUMNS].mean()
    q25_profile = daily_demand[DEMAND_COLUMNS].quantile(0.25)
    q75_profile = daily_demand[DEMAND_COLUMNS].quantile(0.75)
    ax.plot(HOURS, average_profile, color="#1f77b4", linewidth=2.5, label="Promedio")
    ax.fill_between(
        HOURS,
        q25_profile,
        q75_profile,
        color="#1f77b4",
        alpha=0.18,
        label="P25-P75",
    )
    ax.set_title("Perfil horario promedio de demanda")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Demanda (MWh)")
    ax.set_xticks(HOURS)
    ax.grid(alpha=0.25)
    ax.legend()
    return ax


def plot_month_hour_heatmap(daily_demand: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot average hourly demand by month as a heatmap."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.8))
    month_profile = daily_demand.groupby("mes")[DEMAND_COLUMNS].mean()
    image = ax.imshow(month_profile, aspect="auto", cmap="viridis")
    ax.set_title("Demanda promedio por mes y hora")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Mes")
    ax.set_xticks(range(24))
    ax.set_xticklabels(HOURS)
    ax.set_yticks(range(12))
    ax.set_yticklabels(month_profile.index)
    colorbar = ax.figure.colorbar(image, ax=ax)
    colorbar.set_label("Demanda promedio (MWh)")
    return ax


def plot_profile_by_day_type(daily_demand: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot average hourly demand profiles by day type."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    for day_type in DAY_TYPE_ORDER:
        subset = daily_demand[daily_demand["tipo_dia"] == day_type]
        if not subset.empty:
            ax.plot(HOURS, subset[DEMAND_COLUMNS].mean(), linewidth=2, label=day_type)
    ax.set_title("Perfil horario promedio por tipo de dia")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Demanda (MWh)")
    ax.set_xticks(HOURS)
    ax.grid(alpha=0.25)
    ax.legend(title="Tipo de dia")
    return ax


def plot_peak_hour_distribution(
    daily_demand: pd.DataFrame,
    ax: Axes | None = None,
) -> Axes:
    """Plot the distribution of daily peak hours."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    distribution = get_peak_hour_distribution(daily_demand)
    ax.bar(distribution["hora_pico"], distribution["dias"], color="#4c78a8")
    ax.set_title("Distribucion de la hora pico diaria")
    ax.set_xlabel("Hora pico")
    ax.set_ylabel("Cantidad de dias")
    ax.set_xticks(HOURS)
    ax.grid(axis="y", alpha=0.25)
    return ax


def plot_daily_energy_series(daily_demand: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot daily total energy and a 30-day moving average."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ordered_data = daily_demand.sort_values("fecha").copy()
    ordered_data["energia_media_30d_gwh"] = (
        ordered_data["energia_diaria_mwh"].rolling(30, min_periods=7).mean() / 1000
    )
    ax.plot(
        ordered_data["fecha"],
        ordered_data["energia_diaria_mwh"] / 1000,
        color="#9ecae1",
        linewidth=0.8,
        alpha=0.7,
        label="Energia diaria",
    )
    ax.plot(
        ordered_data["fecha"],
        ordered_data["energia_media_30d_gwh"],
        color="#1f77b4",
        linewidth=2,
        label="Media movil 30 dias",
    )
    ax.set_title("Energia diaria total")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Energia diaria (GWh)")
    ax.grid(alpha=0.25)
    ax.legend()
    return ax
