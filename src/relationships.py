import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.shape_level import build_normalized_profiles, fit_pca
from src.daily_demand import DEMAND_COLUMNS

SEASON_MAP = {12: "Verano", 1: "Verano", 2: "Verano",
              3: "Otonio", 4: "Otonio", 5: "Otonio",
              6: "Invierno", 7: "Invierno", 8: "Invierno",
              9: "Primavera", 10: "Primavera", 11: "Primavera"}

MONTH_NAMES = [
    "Ene", "Feb", "Mar", "Abr", "May", "Jun",
    "Jul", "Ago", "Sep", "Oct", "Nov", "Dic",
]


def _merge(daily_demand: pd.DataFrame, conditions: pd.DataFrame) -> pd.DataFrame:
    """Merge daily demand features with condition variables on fecha."""
    return daily_demand.merge(conditions, on="fecha", how="inner", suffixes=("", "_c"))


def _compute_pca_coords(daily_demand: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the first two PCA components of the raw demand profiles
    reusing fit_pca from shape_level.

    Returns a DataFrame with columns pc1, pc2 and stores explained
    variance ratios in attrs.
    """
    pca, scores = fit_pca(daily_demand[DEMAND_COLUMNS], n_components=2)
    result = daily_demand[["fecha"]].copy()
    result["pc1"] = scores[:, 0]
    result["pc2"] = scores[:, 1]
    result.attrs["explained_variance"] = pca.explained_variance_ratio_
    return result


def plot_demand_trend_by_year(
    daily_demand: pd.DataFrame,
    conditions: pd.DataFrame,
) -> None:
    """
    Annual boxplot of daily energy, with the median trend line overlaid.

    Shows whether total demand is growing, stable or declining
    across the 2021-2026 period.
    """
    df = _merge(daily_demand, conditions)
    df["anio"] = df["fecha"].dt.year

    years = sorted(df["anio"].unique())
    data_by_year = [df.loc[df["anio"] == y, "energia_diaria_mwh"].values / 1e3 for y in years]
    medians = [np.median(d) for d in data_by_year]

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.boxplot(
        data_by_year,
        patch_artist=True,
        medianprops={"color": "#1a1a2e", "linewidth": 1.8},
        boxprops={"facecolor": "#4878CF", "alpha": 0.55},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
        widths=0.55,
    )

    ax.plot(range(1, len(years) + 1), medians, marker="o",
            color="#D65F5F", linewidth=1.5, zorder=5, label="mediana anual")

    if 2026 in years:
        idx = years.index(2026) + 1
        ax.axvline(idx, color="#B47CC7", linewidth=1, linestyle="--", alpha=0.7)
        ax.text(idx + 0.05, ax.get_ylim()[0] + 5, "2026 incompleto",
                color="#B47CC7", fontsize=8, va="bottom")

    ax.set_xticks(range(1, len(years) + 1))
    ax.set_xticklabels([str(y) for y in years])
    ax.set_xlabel("Año")
    ax.set_ylabel("Energia diaria (GWh)")
    ax.set_title("Distribucion de energia diaria por año")
    ax.legend()
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    plt.show()


def plot_peak_hour_by_season(
    daily_demand: pd.DataFrame,
    conditions: pd.DataFrame,
) -> None:
    """
    Distribution of the daily peak hour for each season of the year.

    Highlights whether the hour of maximum demand shifts between
    summer (air conditioning) and winter (heating + lighting).
    """
    df = _merge(daily_demand, conditions)
    df["estacion"] = df["fecha"].dt.month.map(SEASON_MAP)

    season_order = ["Verano", "Otonio", "Invierno", "Primavera"]
    colors = ["#D65F5F", "#6ACC65", "#4878CF", "#B47CC7"]

    fig, axes = plt.subplots(1, 4, figsize=(13, 4), sharey=False)

    for ax, season, color in zip(axes, season_order, colors):
        subset = df[df["estacion"] == season]["hora_pico"]
        ax.hist(subset, bins=range(1, 25), color=color, alpha=0.75,
                edgecolor="white", linewidth=0.4)
        ax.axvline(subset.median(), color="#1a1a2e", linewidth=1.5, linestyle="--",
                   label=f"mediana: h{int(subset.median())}")
        ax.set_title(season)
        ax.set_xlabel("Hora del pico")
        ax.legend(fontsize=8)
        ax.set_xlim(1, 24)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    axes[0].set_ylabel("Cantidad de dias")
    fig.suptitle("Distribucion de la hora pico diaria por estacion")
    fig.tight_layout()
    plt.show()


def plot_pca_colored_by(
    daily_demand: pd.DataFrame,
    conditions: pd.DataFrame,
    color_by: str = "mes",
) -> None:
    """
    Scatter of the first two PCA components of raw daily demand profiles,
    colored by a calendar or climate variable.

    Uses fit_pca from shape_level to stay consistent with the rest of the EDA.

    color_by options: 'mes', 'tipo_dia', 'temp_media', 'anio', 'estacion'.
    """
    df = _merge(daily_demand, conditions)
    pca_coords = _compute_pca_coords(daily_demand)
    df = df.merge(pca_coords[["fecha", "pc1", "pc2"]], on="fecha", how="inner")
    explained = pca_coords.attrs.get("explained_variance", [0, 0])

    df["mes"] = df["fecha"].dt.month
    df["anio"] = df["fecha"].dt.year
    df["estacion"] = df["fecha"].dt.month.map(SEASON_MAP)

    if color_by == "temp_media_diaria":
        color_by = "temp_media"

    fig, ax = plt.subplots(figsize=(8, 5))

    if color_by in ("mes", "anio"):
        unique_vals = sorted(df[color_by].unique())
        cmap = plt.colormaps["tab20" if len(unique_vals) > 10 else "tab10"]
        for i, val in enumerate(unique_vals):
            mask = df[color_by] == val
            label = MONTH_NAMES[val - 1] if color_by == "mes" else str(val)
            ax.scatter(df.loc[mask, "pc1"], df.loc[mask, "pc2"],
                       s=10, alpha=0.5, color=cmap(i / len(unique_vals)), label=label)
        ax.legend(title=color_by, fontsize=7, markerscale=2,
                  bbox_to_anchor=(1.01, 1), loc="upper left")

    elif color_by in ("tipo_dia", "estacion"):
        palette = {
            "habil": "#4878CF", "sabado": "#6ACC65",
            "domingo": "#D65F5F", "feriado": "#B47CC7",
            "Verano": "#D65F5F", "Otonio": "#6ACC65",
            "Invierno": "#4878CF", "Primavera": "#B47CC7",
        }
        for val in df[color_by].unique():
            mask = df[color_by] == val
            ax.scatter(df.loc[mask, "pc1"], df.loc[mask, "pc2"],
                       s=10, alpha=0.5, color=palette.get(val, "gray"), label=val)
        ax.legend(title=color_by, fontsize=8, markerscale=2)

    elif color_by == "temp_media":
        sc = ax.scatter(df["pc1"], df["pc2"],
                        c=df["temp_media"], cmap="RdYlBu_r",
                        s=10, alpha=0.6)
        plt.colorbar(sc, ax=ax, label="Temperatura media (C)")

    else:
        ax.scatter(df["pc1"], df["pc2"], s=10, alpha=0.4, color="#4878CF")

    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% varianza)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% varianza)")
    ax.set_title(f"PCA de perfiles diarios — coloreado por {color_by}")
    ax.grid(linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    plt.show()