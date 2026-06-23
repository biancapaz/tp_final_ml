import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

DEMAND_COLUMNS = [f"demanda_{h:02d}" for h in range(1, 25)]
HOURS = list(range(1, 25))

MONTH_NAMES = [
    "Ene", "Feb", "Mar", "Abr", "May", "Jun",
    "Jul", "Ago", "Sep", "Oct", "Nov", "Dic",
]

WEEKDAY_NAMES = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]

SEASON_MAP = {12: "Verano", 1: "Verano", 2: "Verano",
              3: "Otonio", 4: "Otonio", 5: "Otonio",
              6: "Invierno", 7: "Invierno", 8: "Invierno",
              9: "Primavera", 10: "Primavera", 11: "Primavera"}

DAY_TYPE_ORDER = ["habil", "sabado", "domingo", "feriado"]
DAY_TYPE_COLORS = {
    "habil": "#4878CF",
    "sabado": "#6ACC65",
    "domingo": "#D65F5F",
    "feriado": "#B47CC7",
}


def _merge(daily_demand: pd.DataFrame, conditions: pd.DataFrame) -> pd.DataFrame:
    """Merge daily demand features with condition variables on fecha."""
    return daily_demand.merge(conditions, on="fecha", how="inner", suffixes=("", "_c"))


def summarize_by_calendar(
    daily_demand: pd.DataFrame,
    conditions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a summary table of mean daily energy and peak demand
    grouped by month, weekday, and day type.

    Useful to get a quick numeric overview before plotting.
    """
    df = _merge(daily_demand, conditions)
    df["mes"] = df["fecha"].dt.month
    df["dia_semana"] = df["fecha"].dt.dayofweek  # 0=Mon

    by_month = (
        df.groupby("mes")[["energia_diaria_mwh", "demanda_maxima_mwh"]]
        .mean()
        .rename(columns={"energia_diaria_mwh": "energia_media_mwh",
                         "demanda_maxima_mwh": "demanda_max_media_mwh"})
    )
    by_month.index = [MONTH_NAMES[m - 1] for m in by_month.index]
    by_month.index.name = "mes"

    by_type = (
        df.groupby("tipo_dia")[["energia_diaria_mwh", "demanda_maxima_mwh"]]
        .mean()
        .rename(columns={"energia_diaria_mwh": "energia_media_mwh",
                         "demanda_maxima_mwh": "demanda_max_media_mwh"})
    )

    return by_month, by_type


def plot_energy_by_month(
    daily_demand: pd.DataFrame,
    conditions: pd.DataFrame,
) -> None:
    """
    Boxplot of daily energy by calendar month.

    Shows both the distribution and the median to highlight
    the seasonal pattern across the year.
    """
    df = _merge(daily_demand, conditions)
    df["mes"] = df["fecha"].dt.month

    fig, ax = plt.subplots(figsize=(10, 4))

    data_by_month = [df.loc[df["mes"] == m, "energia_diaria_mwh"].values / 1e3
                     for m in range(1, 13)]

    bp = ax.boxplot(
        data_by_month,
        patch_artist=True,
        medianprops={"color": "#1a1a2e", "linewidth": 1.8},
        boxprops={"facecolor": "#4878CF", "alpha": 0.6},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
        widths=0.6,
    )

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_NAMES)
    ax.set_xlabel("Mes")
    ax.set_ylabel("Energia diaria (GWh)")
    ax.set_title("Distribucion de energia diaria por mes")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.grid(axis="y", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    plt.show()


def plot_profile_by_month(
    daily_demand: pd.DataFrame,
    conditions: pd.DataFrame,
) -> None:
    """
    Average hourly demand profile for each calendar month, plotted as a
    grid of 12 subplots.

    Allows comparing how the shape of the daily curve changes across months.
    """
    df = _merge(daily_demand, conditions)
    df["mes"] = df["fecha"].dt.month

    fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharey=True, sharex=True)
    axes_flat = axes.flatten()

    global_max = df[DEMAND_COLUMNS].values.mean(axis=0).max()

    for idx, mes in enumerate(range(1, 13)):
        ax = axes_flat[idx]
        subset = df[df["mes"] == mes]
        mean_profile = subset[DEMAND_COLUMNS].mean()
        p25 = subset[DEMAND_COLUMNS].quantile(0.25)
        p75 = subset[DEMAND_COLUMNS].quantile(0.75)

        ax.fill_between(HOURS, p25 / 1e3, p75 / 1e3, alpha=0.25, color="#4878CF")
        ax.plot(HOURS, mean_profile / 1e3, color="#4878CF", linewidth=1.5)
        ax.set_title(MONTH_NAMES[mes - 1], fontsize=10)
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.grid(linewidth=0.4, alpha=0.5)

    for ax in axes_flat:
        ax.set_xlabel("Hora")
    for ax in axes[:, 0]:
        ax.set_ylabel("Demanda (GWh)")

    fig.suptitle("Perfil horario promedio por mes (banda: p25-p75)", y=1.01)
    fig.tight_layout()
    plt.show()


def plot_energy_by_weekday(
    daily_demand: pd.DataFrame,
    conditions: pd.DataFrame,
) -> None:
    """
    Boxplot of daily energy by day of the week.

    Separates weekdays, Saturday, Sunday and holidays so the
    calendar effect on demand level is visible.
    """
    df = _merge(daily_demand, conditions)
    df["dia_semana"] = df["fecha"].dt.dayofweek  # 0=Lun

    fig, ax = plt.subplots(figsize=(9, 4))

    data_by_day = [
        df.loc[df["dia_semana"] == d, "energia_diaria_mwh"].values / 1e3
        for d in range(7)
    ]

    ax.boxplot(
        data_by_day,
        patch_artist=True,
        medianprops={"color": "#1a1a2e", "linewidth": 1.8},
        boxprops={"facecolor": "#4878CF", "alpha": 0.6},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
        widths=0.6,
    )

    ax.set_xticks(range(1, 8))
    ax.set_xticklabels(WEEKDAY_NAMES)
    ax.set_xlabel("Dia de la semana")
    ax.set_ylabel("Energia diaria (GWh)")
    ax.set_title("Distribucion de energia diaria por dia de la semana")
    ax.grid(axis="y", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    plt.show()


def plot_holiday_vs_workday(
    daily_demand: pd.DataFrame,
    conditions: pd.DataFrame,
) -> None:
    """
    Compare average hourly profiles for workdays, Saturdays, Sundays and
    holidays on the same axes.

    Useful to decide if holidays should be modeled as a separate category
    or grouped with Sundays.
    """
    df = _merge(daily_demand, conditions)

    fig, ax = plt.subplots(figsize=(9, 4))

    for day_type in DAY_TYPE_ORDER:
        subset = df[df["tipo_dia"] == day_type]
        if subset.empty:
            continue
        mean_profile = subset[DEMAND_COLUMNS].mean() / 1e3
        ax.plot(
            HOURS,
            mean_profile,
            label=day_type.capitalize(),
            color=DAY_TYPE_COLORS[day_type],
            linewidth=1.8,
        )

    ax.set_xlabel("Hora")
    ax.set_ylabel("Demanda media (GWh)")
    ax.set_title("Perfil horario promedio por tipo de dia")
    ax.set_xticks([1, 6, 12, 18, 24])
    ax.legend()
    ax.grid(linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    plt.show()
