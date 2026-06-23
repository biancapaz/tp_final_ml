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


def _merge(daily_demand: pd.DataFrame, conditions: pd.DataFrame) -> pd.DataFrame:
    """Merge daily demand features with condition variables on fecha."""
    return daily_demand.merge(conditions, on="fecha", how="inner", suffixes=("", "_c"))


def summarize_climate_demand_correlation(
    daily_demand: pd.DataFrame,
    conditions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return Pearson correlations between daily energy / peak demand
    and each available climate variable.

    Higher absolute correlation indicates a more informative conditioning variable.
    """
    df = _merge(daily_demand, conditions)

    climate_vars = [c for c in conditions.columns
                    if c not in ("fecha",) and conditions[c].dtype != object]
    demand_vars = ["energia_diaria_mwh", "demanda_maxima_mwh"]

    rows = []
    for cv in climate_vars:
        row = {"variable_clima": cv}
        for dv in demand_vars:
            try:
                row[dv] = df[[cv, dv]].corr().iloc[0, 1]
            except Exception:
                row[dv] = float("nan")
        rows.append(row)

    result = pd.DataFrame(rows).set_index("variable_clima")
    return result.round(3).sort_values("energia_diaria_mwh", key=abs, ascending=False)


def plot_climate_by_month(conditions: pd.DataFrame) -> None:
    """
    Plot monthly distributions of temperature, apparent temperature,
    humidity and wind speed.

    Shows how the climate regime varies across the year,
    which helps interpret demand seasonality.
    """
    df = conditions.copy()
    df["mes"] = df["fecha"].dt.month

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharey=False)

    climate_specs = [
        ("temp_media", "Temperatura media (C)", "#D65F5F"),
        ("sensacion_media", "Sensacion termica media (C)", "#B47CC7"),
        ("humedad_media", "Humedad media (%)", "#4878CF"),
        ("viento_media", "Viento medio (km/h)", "#6ACC65"),
    ]

    for ax, (col, label, color) in zip(axes.flatten(), climate_specs):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        # Use daily aggregated values: mean per day, then box by month
        data_by_month = [df.loc[df["mes"] == m, col].values for m in range(1, 13)]
        ax.boxplot(
            data_by_month,
            patch_artist=True,
            medianprops={"color": "#1a1a2e", "linewidth": 1.5},
            boxprops={"facecolor": color, "alpha": 0.5},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
            flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.35},
            widths=0.6,
        )
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_NAMES, fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    fig.suptitle("Distribucion de variables climaticas por mes")
    fig.tight_layout()
    plt.show()


def plot_demand_vs_temperature(
    daily_demand: pd.DataFrame,
    conditions: pd.DataFrame,
) -> None:
    """
    Scatter plot of daily energy vs daily mean effective temperature,
    with a smoothed LOWESS trend line.

    A U-shaped curve would indicate that both heat and cold drive demand up.
    """
    df = _merge(daily_demand, conditions)

    if "temp_media" not in df.columns:
        print("Columna 'temp_media' no encontrada en conditions.")
        return

    # Compute daily mean temperature from hourly if not already aggregated
    temp_col = "temp_media"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: energia diaria vs temp media
    ax = axes[0]
    ax.scatter(df[temp_col], df["energia_diaria_mwh"] / 1e3,
               alpha=0.25, s=12, color="#4878CF")

    # Smoothed trend via rolling on sorted values
    sorted_df = df[[temp_col, "energia_diaria_mwh"]].dropna().sort_values(temp_col)
    window = max(30, len(sorted_df) // 20)
    trend = sorted_df["energia_diaria_mwh"].rolling(window, center=True, min_periods=10).mean()
    ax.plot(sorted_df[temp_col], trend / 1e3, color="#D65F5F", linewidth=2, label="tendencia")

    ax.set_xlabel("Temperatura media media diaria (C)")
    ax.set_ylabel("Energia diaria (GWh)")
    ax.set_title("Energia diaria vs temperatura")
    ax.legend()
    ax.grid(linewidth=0.4, alpha=0.5)

    # Right: demanda maxima vs temp
    ax = axes[1]
    ax.scatter(df[temp_col], df["demanda_maxima_mwh"] / 1e3,
               alpha=0.25, s=12, color="#6ACC65")

    sorted_df2 = df[[temp_col, "demanda_maxima_mwh"]].dropna().sort_values(temp_col)
    trend2 = sorted_df2["demanda_maxima_mwh"].rolling(window, center=True, min_periods=10).mean()
    ax.plot(sorted_df2[temp_col], trend2 / 1e3, color="#D65F5F", linewidth=2, label="tendencia")

    ax.set_xlabel("Temperatura media media diaria (C)")
    ax.set_ylabel("Demanda maxima (GWh)")
    ax.set_title("Demanda maxima diaria vs temperatura")
    ax.legend()
    ax.grid(linewidth=0.4, alpha=0.5)

    fig.suptitle("Relacion entre temperatura y demanda")
    fig.tight_layout()
    plt.show()


def plot_demand_vs_apparent_temp(
    daily_demand: pd.DataFrame,
    conditions: pd.DataFrame,
) -> None:
    """
    Compare the demand-vs-temperature relationship using real temperature
    and apparent temperature (sensacion termica) side by side.

    If apparent temperature has a tighter relationship, it is a better
    conditioning variable for the model.
    """
    df = _merge(daily_demand, conditions)

    missing = [c for c in ("temp_media", "sensacion_media") if c not in df.columns]
    if missing:
        print(f"Columnas faltantes: {missing}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, col, label, color in [
        (axes[0], "temp_media", "Temperatura media (C)", "#4878CF"),
        (axes[1], "sensacion_media", "Sensacion termica media (C)", "#B47CC7"),
    ]:
        ax.scatter(df[col], df["energia_diaria_mwh"] / 1e3,
                   alpha=0.25, s=12, color=color)

        sorted_df = df[[col, "energia_diaria_mwh"]].dropna().sort_values(col)
        window = max(30, len(sorted_df) // 20)
        trend = sorted_df["energia_diaria_mwh"].rolling(window, center=True, min_periods=10).mean()
        ax.plot(sorted_df[col], trend / 1e3, color="#D65F5F", linewidth=2)

        corr = df[[col, "energia_diaria_mwh"]].corr().iloc[0, 1]
        ax.set_xlabel(label)
        ax.set_title(f"r = {corr:.3f}")
        ax.grid(linewidth=0.4, alpha=0.5)

    axes[0].set_ylabel("Energia diaria (GWh)")
    fig.suptitle("Temperatura real vs sensacion termica como predictor de demanda")
    fig.tight_layout()
    plt.show()


def plot_profile_by_temperature_tercile(
    daily_demand: pd.DataFrame,
    conditions: pd.DataFrame,
) -> None:
    """
    Average hourly profile for cold, mild and hot days, defined by
    terciles of the daily mean effective temperature.

    Shows whether temperature affects the shape of the profile or only its level.
    """
    df = _merge(daily_demand, conditions)

    if "temp_media" not in df.columns:
        print("Columna 'temp_media' no encontrada.")
        return

    terciles = df["temp_media"].quantile([1/3, 2/3])
    t1, t2 = terciles.iloc[0], terciles.iloc[1]

    groups = {
        f"Frio (< {t1:.1f} C)": df["temp_media"] < t1,
        f"Templado ({t1:.1f}–{t2:.1f} C)": (df["temp_media"] >= t1) & (df["temp_media"] < t2),
        f"Caluroso (> {t2:.1f} C)": df["temp_media"] >= t2,
    }

    colors = ["#4878CF", "#6ACC65", "#D65F5F"]

    fig, ax = plt.subplots(figsize=(9, 4))

    for (label, mask), color in zip(groups.items(), colors):
        subset = df[mask]
        mean_profile = subset[DEMAND_COLUMNS].mean() / 1e3
        p25 = subset[DEMAND_COLUMNS].quantile(0.25) / 1e3
        p75 = subset[DEMAND_COLUMNS].quantile(0.75) / 1e3
        ax.fill_between(HOURS, p25, p75, alpha=0.15, color=color)
        ax.plot(HOURS, mean_profile, color=color, linewidth=1.8, label=f"{label} (n={mask.sum()})")

    ax.set_xlabel("Hora")
    ax.set_ylabel("Demanda media (GWh)")
    ax.set_title("Perfil horario promedio por tercil de temperatura diaria")
    ax.set_xticks([1, 6, 12, 18, 24])
    ax.legend(fontsize=9)
    ax.grid(linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    plt.show()