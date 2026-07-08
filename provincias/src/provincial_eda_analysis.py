from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROVINCE_ORDER = [
    "buenos_aires", "catamarca", "chaco", "chubut", "cordoba",
    "corrientes", "entre_rios", "formosa", "jujuy", "la_pampa",
    "la_rioja", "mendoza", "misiones", "neuquen", "rio_negro",
    "salta", "san_juan", "san_luis", "santa_cruz", "santa_fe",
    "sgo_del_estero", "tucuman",
]
HOURS = list(range(1, 25))
MONTH_NAMES = [
    "Ene", "Feb", "Mar", "Abr", "May", "Jun",
    "Jul", "Ago", "Sep", "Oct", "Nov", "Dic",
]
SEASON_MAP = {
    12: "Verano", 1: "Verano", 2: "Verano",
    3: "Otonio", 4: "Otonio", 5: "Otonio",
    6: "Invierno", 7: "Invierno", 8: "Invierno",
    9: "Primavera", 10: "Primavera", 11: "Primavera",
}
PROVINCE_DISPLAY_NAMES = {
    "buenos_aires": "Buenos Aires",
    "catamarca": "Catamarca",
    "chaco": "Chaco",
    "chubut": "Chubut",
    "cordoba": "Córdoba",
    "corrientes": "Corrientes",
    "entre_rios": "Entre Ríos",
    "formosa": "Formosa",
    "jujuy": "Jujuy",
    "la_pampa": "La Pampa",
    "la_rioja": "La Rioja",
    "mendoza": "Mendoza",
    "misiones": "Misiones",
    "neuquen": "Neuquén",
    "rio_negro": "Río Negro",
    "salta": "Salta",
    "san_juan": "San Juan",
    "san_luis": "San Luis",
    "santa_cruz": "Santa Cruz",
    "santa_fe": "Santa Fe",
    "sgo_del_estero": "Santiago del Estero",
    "tucuman": "Tucumán",
}
REPORT_PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#E69F00",
    "#CC79A7",
    "#56B4E9",
]


def format_province_name(province: str) -> str:
    """Return a readable province name for report figures."""
    return PROVINCE_DISPLAY_NAMES.get(province, province.replace("_", " ").title())


def save_report_figure(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    """Save a report figure with high-quality export defaults."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return output_path


def get_daily_energy_by_province(hourly_demand: pd.DataFrame) -> pd.DataFrame:
    """Build a long daily province-level energy table in MWh and GWh."""
    daily = build_daily_province_energy(hourly_demand)
    daily["energia_diaria_gwh"] = daily["energia_diaria_mwh"] / 1000
    daily["provincia_nombre"] = daily["provincia"].map(format_province_name)
    return daily


def get_normalized_hourly_profile(
    hourly_demand: pd.DataFrame,
    provinces: list[str],
) -> pd.DataFrame:
    """Compute average hourly profiles normalized by each province's own daily mean."""
    profiles = hourly_demand.groupby("hora")[provinces].mean().reindex(HOURS)
    return profiles.div(profiles.mean(axis=0), axis=1)


def _style_report_axis(ax: Axes, grid_axis: str = "y") -> None:
    """Apply a compact report style to one axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.22, linewidth=0.8)


def compute_hourly_shares(hourly_demand: pd.DataFrame) -> pd.DataFrame:
    """Compute hourly provincial shares as percentages of the provincial total."""
    total = hourly_demand[PROVINCE_ORDER].sum(axis=1)
    shares = hourly_demand[["fecha", "hora", "mes", "dia_semana", "es_feriado"]].copy()
    shares[PROVINCE_ORDER] = hourly_demand[PROVINCE_ORDER].div(total, axis=0) * 100
    return shares


def build_daily_province_energy(hourly_demand: pd.DataFrame) -> pd.DataFrame:
    """Build a long daily province-level energy table in MWh."""
    daily_wide = hourly_demand.groupby("fecha")[PROVINCE_ORDER].sum(min_count=1)
    daily = daily_wide.reset_index().melt(
        id_vars="fecha",
        var_name="provincia",
        value_name="energia_diaria_mwh",
    )
    daily["mes"] = daily["fecha"].dt.month
    daily["estacion"] = daily["mes"].map(SEASON_MAP)
    return daily


def summarize_share_variability(hourly_demand: pd.DataFrame) -> pd.DataFrame:
    """Summarize how stable each province's hourly share is."""
    shares = compute_hourly_shares(hourly_demand)
    rows = []
    for province in PROVINCE_ORDER:
        values = shares[province]
        rows.append(
            {
                "provincia": province,
                "share_medio_pct": values.mean(),
                "share_std_pct": values.std(),
                "share_p05_pct": values.quantile(0.05),
                "share_p95_pct": values.quantile(0.95),
                "rango_p90_pct": values.quantile(0.95) - values.quantile(0.05),
                "cv_share": values.std() / values.mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("rango_p90_pct", ascending=False)


def compare_shares_on_extreme_days(hourly_demand: pd.DataFrame) -> pd.DataFrame:
    """Compare average shares on the highest and lowest national demand days."""
    daily_total = hourly_demand.assign(
        total=hourly_demand[PROVINCE_ORDER].sum(axis=1)
    ).groupby("fecha")["total"].sum()
    high_date = daily_total.idxmax()
    low_date = daily_total.idxmin()
    shares = compute_hourly_shares(hourly_demand)
    regular = shares[PROVINCE_ORDER].mean()
    high = shares.loc[shares["fecha"] == high_date, PROVINCE_ORDER].mean()
    low = shares.loc[shares["fecha"] == low_date, PROVINCE_ORDER].mean()
    result = pd.DataFrame(
        {
            "provincia": PROVINCE_ORDER,
            "share_promedio_pct": regular.values,
            f"share_max_demanda_{high_date.date()}": high.values,
            f"share_min_demanda_{low_date.date()}": low.values,
        }
    )
    result["dif_max_vs_promedio"] = result[f"share_max_demanda_{high_date.date()}"] - result[
        "share_promedio_pct"
    ]
    result["dif_min_vs_promedio"] = result[f"share_min_demanda_{low_date.date()}"] - result[
        "share_promedio_pct"
    ]
    return result.reindex(result["dif_max_vs_promedio"].abs().sort_values(ascending=False).index)


def plot_share_variability(summary: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot the P5-P95 share range by province."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))
    ordered = summary.sort_values("rango_p90_pct", ascending=False)
    ax.bar(ordered["provincia"], ordered["rango_p90_pct"], color="#4c78a8")
    ax.set_title("Variabilidad del share horario provincial")
    ax.set_xlabel("Provincia")
    ax.set_ylabel("Rango P5-P95 del share (p.p.)")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(axis="y", alpha=0.25)
    return ax


def plot_absolute_vs_share_profile(
    hourly_demand: pd.DataFrame,
    province: str,
    ax: Axes | None = None,
) -> Axes:
    """Compare the average hourly demand profile against the average hourly share profile."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    profile_mwh = hourly_demand.groupby("hora")[province].mean()
    shares = compute_hourly_shares(hourly_demand)
    profile_share = shares.groupby("hora")[province].mean()
    ax.plot(profile_mwh.index, profile_mwh / profile_mwh.mean(), label="MWh normalizado")
    ax.plot(profile_share.index, profile_share / profile_share.mean(), label="Share normalizado")
    ax.set_title(f"Perfil absoluto vs share — {province}")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Indice promedio = 1")
    ax.set_xticks(HOURS)
    ax.grid(alpha=0.25)
    ax.legend()
    return ax


def compute_daily_energy_correlation(hourly_demand: pd.DataFrame) -> pd.DataFrame:
    """Compute the correlation matrix between daily province-level energy series."""
    return hourly_demand.groupby("fecha")[PROVINCE_ORDER].sum(min_count=1).corr()


def summarize_province_correlations(correlation: pd.DataFrame) -> pd.DataFrame:
    """Summarize each province's average correlation with the rest of the system."""
    rows = []
    for province in PROVINCE_ORDER:
        values = correlation.loc[province].drop(province)
        rows.append(
            {
                "provincia": province,
                "correlacion_media_con_resto": values.mean(),
                "correlacion_minima": values.min(),
                "provincia_menos_correlacionada": values.idxmin(),
                "correlacion_maxima": values.max(),
                "provincia_mas_correlacionada": values.idxmax(),
            }
        )
    return pd.DataFrame(rows).sort_values("correlacion_media_con_resto", ascending=False)


def plot_correlation_heatmap(correlation: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot a province-by-province correlation heatmap."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(correlation.loc[PROVINCE_ORDER, PROVINCE_ORDER], vmin=0, vmax=1, cmap="viridis")
    ax.set_title("Correlacion entre energias diarias provinciales")
    ax.set_xticks(range(len(PROVINCE_ORDER)))
    ax.set_xticklabels(PROVINCE_ORDER, rotation=90)
    ax.set_yticks(range(len(PROVINCE_ORDER)))
    ax.set_yticklabels(PROVINCE_ORDER)
    colorbar = ax.figure.colorbar(image, ax=ax)
    colorbar.set_label("Correlacion")
    return ax


def plot_province_scatter(
    hourly_demand: pd.DataFrame,
    province_x: str,
    province_y: str,
    ax: Axes | None = None,
) -> Axes:
    """Plot daily energy for two provinces colored by month."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    daily = hourly_demand.groupby("fecha")[PROVINCE_ORDER].sum(min_count=1).reset_index()
    daily["mes"] = daily["fecha"].dt.month
    scatter = ax.scatter(daily[province_x], daily[province_y], c=daily["mes"], cmap="viridis", s=18)
    ax.set_title(f"Energia diaria: {province_x} vs {province_y}")
    ax.set_xlabel(f"{province_x} (MWh/dia)")
    ax.set_ylabel(f"{province_y} (MWh/dia)")
    colorbar = ax.figure.colorbar(scatter, ax=ax)
    colorbar.set_label("Mes")
    ax.grid(alpha=0.25)
    return ax


def summarize_heterogeneity(hourly_demand: pd.DataFrame) -> pd.DataFrame:
    """Summarize scale, variability and peak-hour behavior for each province."""
    daily_wide = hourly_demand.groupby("fecha")[PROVINCE_ORDER].sum(min_count=1)
    rows = []
    for province in PROVINCE_ORDER:
        hourly = hourly_demand[["fecha", "hora", province]].copy()
        daily_profiles = hourly.pivot(index="fecha", columns="hora", values=province)
        peak_hours = daily_profiles.idxmax(axis=1)
        min_values = daily_profiles.min(axis=1).replace(0, np.nan)
        amplitude_ratio = daily_profiles.max(axis=1) / min_values
        rows.append(
            {
                "provincia": province,
                "energia_media_diaria_mwh": daily_wide[province].mean(),
                "energia_mediana_diaria_mwh": daily_wide[province].median(),
                "energia_std_diaria_mwh": daily_wide[province].std(),
                "cv_energia_diaria": daily_wide[province].std() / daily_wide[province].mean(),
                "ratio_max_min_promedio": amplitude_ratio.mean(),
                "hora_pico_modal": int(peak_hours.mode().iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("cv_energia_diaria", ascending=False)


def plot_daily_energy_boxplot(hourly_demand: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot daily energy distributions by province in GWh."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))
    daily = hourly_demand.groupby("fecha")[PROVINCE_ORDER].sum(min_count=1) / 1000
    order = daily.median().sort_values(ascending=False).index.tolist()
    ax.boxplot([daily[province].dropna() for province in order], labels=order, showfliers=False)
    ax.set_title("Distribucion de energia diaria por provincia")
    ax.set_xlabel("Provincia")
    ax.set_ylabel("Energia diaria (GWh)")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(axis="y", alpha=0.25)
    return ax


def plot_report_daily_energy_boxplot(
    hourly_demand: pd.DataFrame,
    output_dir: Path | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot a report-ready daily energy boxplot by province."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12.5, 5.6))

    daily = hourly_demand.groupby("fecha")[PROVINCE_ORDER].sum(min_count=1) / 1000
    order = daily.median().sort_values(ascending=False).index.tolist()
    labels = [format_province_name(province) for province in order]

    ax.boxplot(
        [daily[province].dropna() for province in order],
        tick_labels=labels,
        showfliers=False,
        patch_artist=True,
        widths=0.62,
        boxprops={"facecolor": "#D8E4EF", "edgecolor": "#355C7D", "linewidth": 1.0},
        medianprops={"color": "#1D3557", "linewidth": 1.4},
        whiskerprops={"color": "#516A7D", "linewidth": 0.9},
        capprops={"color": "#516A7D", "linewidth": 0.9},
    )
    ax.set_xlabel("Provincia")
    ax.set_ylabel("Energía diaria (GWh)")
    ax.tick_params(axis="x", rotation=75, labelsize=8)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    _style_report_axis(ax, grid_axis="y")
    ax.figure.tight_layout()

    if output_dir is not None:
        save_report_figure(ax.figure, output_dir, "eda_daily_energy_boxplot_by_province.png")
    return ax


def plot_report_daily_energy_boxplot_log(
    hourly_demand: pd.DataFrame,
    output_dir: Path | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot a horizontal log-scale daily energy boxplot by province."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9.5, 7.2))

    daily = hourly_demand.groupby("fecha")[PROVINCE_ORDER].sum(min_count=1) / 1000
    order = daily.median().sort_values(ascending=False).index.tolist()
    labels = [format_province_name(province) for province in order]
    values = []
    for province in order:
        province_values = daily[province].dropna()
        values.append(province_values.loc[province_values > 0])

    ax.boxplot(
        values,
        vert=False,
        tick_labels=labels,
        showfliers=False,
        patch_artist=True,
        widths=0.58,
        boxprops={"facecolor": "#D8E4EF", "edgecolor": "#355C7D", "linewidth": 1.0},
        medianprops={"color": "#1D3557", "linewidth": 1.4},
        whiskerprops={"color": "#516A7D", "linewidth": 0.9},
        capprops={"color": "#516A7D", "linewidth": 0.9},
    )
    ax.set_xscale("log")
    tick_candidates = np.array([1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300])
    x_min, x_max = ax.get_xlim()
    ticks = tick_candidates[(tick_candidates >= x_min) & (tick_candidates <= x_max)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick:g}" for tick in ticks])
    ax.set_xlabel("Energía diaria (GWh, escala log)")
    ax.set_ylabel("Provincia")
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=9)
    _style_report_axis(ax, grid_axis="x")
    ax.figure.tight_layout()

    if output_dir is not None:
        save_report_figure(
            ax.figure,
            output_dir,
            "eda_daily_energy_boxplot_by_province_log_horizontal.png",
        )
    return ax


def plot_report_normalized_hourly_profiles(
    hourly_demand: pd.DataFrame,
    provinces: list[str] | None = None,
    output_dir: Path | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot normalized hourly profiles for selected provinces in one axis."""
    if provinces is None:
        provinces = ["buenos_aires", "cordoba", "santa_fe", "formosa", "neuquen", "santa_cruz"]
    if ax is None:
        _, ax = plt.subplots(figsize=(8.8, 4.8))

    profiles = get_normalized_hourly_profile(hourly_demand, provinces)
    for province, color in zip(provinces, REPORT_PALETTE):
        ax.plot(
            HOURS,
            profiles[province],
            color=color,
            linewidth=2.1,
            label=format_province_name(province),
        )

    ax.set_xlabel("Hora")
    ax.set_ylabel("Demanda horaria / media diaria provincial")
    ax.set_xlim(1, 24)
    ax.set_xticks(HOURS)
    ax.set_ylim(0.68, 1.32)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _style_report_axis(ax, grid_axis="both")
    ax.figure.tight_layout()

    if output_dir is not None:
        save_report_figure(
            ax.figure,
            output_dir,
            "eda_normalized_hourly_profiles_selected_provinces.png",
        )
    return ax


def _compute_temperature_demand_table(
    hourly_demand: pd.DataFrame,
    climate_daily: pd.DataFrame,
    provinces: list[str],
) -> pd.DataFrame:
    """Build normalized daily energy and temperature data for selected provinces."""
    daily = get_daily_energy_by_province(hourly_demand)
    climate = climate_daily.copy()
    climate["fecha"] = pd.to_datetime(climate["fecha"])
    merged = daily.merge(
        climate[["fecha", "provincia", "temp_media"]],
        on=["fecha", "provincia"],
        how="inner",
    )
    merged = merged.loc[merged["provincia"].isin(provinces)].copy()
    mean_energy = merged.groupby("provincia")["energia_diaria_mwh"].transform("mean")
    merged["energia_diaria_normalizada"] = merged["energia_diaria_mwh"] / mean_energy
    return merged


def _add_binned_mean_line(
    ax: Axes,
    data: pd.DataFrame,
    color: str,
    bins: int = 18,
) -> None:
    """Add a smooth binned-mean line to a temperature-demand scatter plot."""
    clean = data[["temp_media", "energia_diaria_normalizada"]].dropna()
    if clean["temp_media"].nunique() < 3:
        return

    edges = np.linspace(clean["temp_media"].min(), clean["temp_media"].max(), bins + 1)
    edges = np.unique(edges)
    if len(edges) < 3:
        return

    binned = (
        clean.assign(temp_bin=pd.cut(clean["temp_media"], bins=edges, include_lowest=True))
        .groupby("temp_bin", observed=True)
        .agg(
            temp_media=("temp_media", "mean"),
            energia_diaria_normalizada=("energia_diaria_normalizada", "mean"),
            observaciones=("energia_diaria_normalizada", "size"),
        )
        .query("observaciones >= 5")
    )
    if len(binned) < 2:
        return

    ax.plot(
        binned["temp_media"],
        binned["energia_diaria_normalizada"],
        color=color,
        linewidth=2.5,
    )


def plot_report_temperature_demand_scatter(
    hourly_demand: pd.DataFrame,
    climate_daily: pd.DataFrame,
    provinces: list[str] | None = None,
    output_dir: Path | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot normalized daily demand against daily mean temperature for selected provinces."""
    if provinces is None:
        provinces = ["formosa", "buenos_aires", "neuquen"]
    if ax is None:
        _, ax = plt.subplots(figsize=(8.4, 5.2))

    merged = _compute_temperature_demand_table(hourly_demand, climate_daily, provinces)
    colors = REPORT_PALETTE[: len(provinces)]
    legend_handles = []
    for province, color in zip(provinces, colors):
        data = merged.loc[merged["provincia"] == province]
        ax.scatter(
            data["temp_media"],
            data["energia_diaria_normalizada"],
            s=14,
            alpha=0.18,
            color=color,
            edgecolors="none",
        )
        _add_binned_mean_line(ax, data, color=color)
        legend_handles.append(
            Line2D([0], [0], color=color, linewidth=3, label=format_province_name(province))
        )

    ax.axhline(1, color="#5F6F7E", linewidth=0.9, linestyle="--", alpha=0.65)
    ax.set_xlabel("Temperatura media diaria (°C)")
    ax.set_ylabel("Energía diaria / media provincial")
    ax.legend(handles=legend_handles, frameon=False)
    _style_report_axis(ax, grid_axis="both")
    ax.figure.tight_layout()

    if output_dir is not None:
        save_report_figure(
            ax.figure,
            output_dir,
            "eda_temperature_demand_scatter_selected_provinces.png",
        )
    return ax


def _compute_weekday_sunday_difference(hourly_demand: pd.DataFrame) -> pd.DataFrame:
    """Compute hourly weekday-minus-Sunday differences relative to mean daily energy."""
    regular_weekday = (hourly_demand["dia_semana"].isin([0, 1, 2, 3, 4])) & (
        hourly_demand["es_feriado"] == 0
    )
    sunday = hourly_demand["dia_semana"] == 6
    weekday_profile = hourly_demand.loc[regular_weekday].groupby("hora")[PROVINCE_ORDER].mean()
    sunday_profile = hourly_demand.loc[sunday].groupby("hora")[PROVINCE_ORDER].mean()
    mean_daily_energy = hourly_demand.groupby("fecha")[PROVINCE_ORDER].sum(min_count=1).mean()
    difference = (weekday_profile - sunday_profile).div(mean_daily_energy, axis=1).T
    difference = difference.reindex(index=PROVINCE_ORDER, columns=HOURS)
    order = difference.sum(axis=1).sort_values(ascending=False).index.tolist()
    return difference.loc[order]


def plot_report_weekday_sunday_difference_heatmap(
    hourly_demand: pd.DataFrame,
    output_dir: Path | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot weekday-minus-Sunday hourly profile differences by province."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10.8, 8.0))

    difference = _compute_weekday_sunday_difference(hourly_demand)
    limit = float(np.nanmax(np.abs(difference.to_numpy())))
    if limit == 0 or np.isnan(limit):
        limit = 1.0

    image = ax.imshow(
        difference,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    ax.set_xlabel("Hora")
    ax.set_ylabel("Provincia")
    ax.set_xticks(range(len(HOURS)))
    ax.set_xticklabels(HOURS)
    ax.set_yticks(range(len(difference.index)))
    ax.set_yticklabels([format_province_name(province) for province in difference.index])
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Diferencia relativa hábil - domingo")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    ax.figure.tight_layout()

    if output_dir is not None:
        save_report_figure(ax.figure, output_dir, "eda_weekday_sunday_difference_heatmap.png")
    return ax


def plot_mean_profiles_grid(hourly_demand: pd.DataFrame) -> plt.Figure:
    """Plot average hourly profiles for all provinces in a compact grid."""
    fig, axes = plt.subplots(6, 4, figsize=(12, 12), sharex=True)
    profiles = hourly_demand.groupby("hora")[PROVINCE_ORDER].mean()
    for ax, province in zip(axes.flatten(), PROVINCE_ORDER):
        profile = profiles[province]
        ax.plot(profile.index, profile / profile.mean(), color="#4c78a8")
        ax.set_title(province, fontsize=9)
        ax.set_ylim(0.65, 1.35)
        ax.grid(alpha=0.2)
    for ax in axes.flatten()[len(PROVINCE_ORDER):]:
        ax.axis("off")
    fig.suptitle("Perfiles horarios medios normalizados por provincia", y=0.995)
    fig.supxlabel("Hora")
    fig.supylabel("Indice promedio = 1")
    fig.tight_layout()
    return fig


def build_socioeconomic_demand_table(
    hourly_demand: pd.DataFrame,
    indicators: pd.DataFrame,
) -> pd.DataFrame:
    """Merge average provincial demand with socioeconomic indicators."""
    daily = hourly_demand.groupby("fecha")[PROVINCE_ORDER].sum(min_count=1)
    mean_demand = daily.mean().rename("energia_media_diaria_mwh").reset_index()
    mean_demand = mean_demand.rename(columns={"index": "provincia"})
    table = indicators.merge(mean_demand, on="provincia", how="left")
    table["consumo_mwh_por_habitante_dia"] = table["energia_media_diaria_mwh"] / table["poblacion"]
    return table


def summarize_socioeconomic_correlations(table: pd.DataFrame) -> pd.DataFrame:
    """Compute correlations between socioeconomic indicators and average demand."""
    columns = [
        "energia_media_diaria_mwh",
        "poblacion",
        "pbg_per_capita",
        "participacion_industrial",
        "superficie_km2",
        "consumo_mwh_por_habitante_dia",
    ]
    return table[columns].corr()


def plot_socioeconomic_scatter(
    table: pd.DataFrame,
    x: str,
    y: str = "energia_media_diaria_mwh",
    ax: Axes | None = None,
) -> Axes:
    """Plot provincial average demand against one socioeconomic indicator."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(table[x], table[y], color="#4c78a8")
    for _, row in table.iterrows():
        ax.annotate(row["provincia"], (row[x], row[y]), fontsize=8, alpha=0.8)
    ax.set_title(f"Demanda promedio vs {x}")
    ax.set_xlabel(x)
    ax.set_ylabel("Energia media diaria (MWh)")
    ax.grid(alpha=0.25)
    return ax


def compute_monthly_temperature(climate_daily: pd.DataFrame) -> pd.DataFrame:
    """Compute average monthly temperature by province."""
    climate = climate_daily.copy()
    climate["fecha"] = pd.to_datetime(climate["fecha"])
    climate["mes"] = climate["fecha"].dt.month
    return climate.pivot_table(
        index="provincia",
        columns="mes",
        values="temp_media",
        aggfunc="mean",
    ).reindex(PROVINCE_ORDER)


def summarize_climate_demand_by_province(
    hourly_demand: pd.DataFrame,
    climate_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Compute province-level correlations between climate variables and daily demand."""
    daily = build_daily_province_energy(hourly_demand)
    climate = climate_daily.copy()
    climate["fecha"] = pd.to_datetime(climate["fecha"])
    merged = daily.merge(climate, on=["fecha", "provincia"], how="inner")
    rows = []
    for province, data in merged.groupby("provincia"):
        rows.append(
            {
                "provincia": province,
                "corr_temp_media": data["energia_diaria_mwh"].corr(data["temp_media"]),
                "corr_grado_calor": data["energia_diaria_mwh"].corr(data["grado_calor"]),
                "corr_grado_frio": data["energia_diaria_mwh"].corr(data["grado_frio"]),
                "dias_con_calor": int((data["grado_calor"] > 0).sum()),
                "dias_con_frio": int((data["grado_frio"] > 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("corr_grado_calor", ascending=False)


def plot_temperature_heatmap(monthly_temperature: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot a province-by-month temperature heatmap."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(monthly_temperature, aspect="auto", cmap="coolwarm")
    ax.set_title("Temperatura media por provincia y mes")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Provincia")
    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTH_NAMES)
    ax.set_yticks(range(len(monthly_temperature.index)))
    ax.set_yticklabels(monthly_temperature.index)
    colorbar = ax.figure.colorbar(image, ax=ax)
    colorbar.set_label("Temperatura media (C)")
    return ax


def plot_climate_demand_examples(
    hourly_demand: pd.DataFrame,
    climate_daily: pd.DataFrame,
    provinces: list[str],
) -> plt.Figure:
    """Plot temperature-demand relationships for selected provinces."""
    daily = build_daily_province_energy(hourly_demand)
    climate = climate_daily.copy()
    climate["fecha"] = pd.to_datetime(climate["fecha"])
    merged = daily.merge(climate, on=["fecha", "provincia"], how="inner")
    fig, axes = plt.subplots(1, len(provinces), figsize=(5 * len(provinces), 4), sharey=False)
    if len(provinces) == 1:
        axes = [axes]
    for ax, province in zip(axes, provinces):
        data = merged.loc[merged["provincia"] == province]
        ax.scatter(data["temp_media"], data["energia_diaria_mwh"], s=12, alpha=0.35, color="#4c78a8")
        ax.set_title(province)
        ax.set_xlabel("Temperatura media (C)")
        ax.set_ylabel("Energia diaria (MWh)")
        ax.grid(alpha=0.25)
    fig.suptitle("Relacion temperatura-demanda en provincias seleccionadas", y=1.03)
    fig.tight_layout()
    return fig


def run_provincial_pca(
    x_profiles: pd.DataFrame,
    n_components: int = 12,
    scale: bool = False,
) -> dict[str, object]:
    """Run PCA on the provincial daily profile matrix."""
    feature_columns = [column for column in x_profiles.columns if column != "fecha"]
    x = x_profiles[feature_columns].copy()
    x = x.fillna(x.median(numeric_only=True))
    matrix = x.to_numpy()
    if scale:
        matrix = StandardScaler().fit_transform(matrix)
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(matrix)
    explained = pd.DataFrame(
        {
            "componente": [f"PC{i + 1}" for i in range(n_components)],
            "varianza_explicada": pca.explained_variance_ratio_,
            "varianza_acumulada": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    scores_df = pd.DataFrame(scores[:, :3], columns=["PC1", "PC2", "PC3"])
    scores_df["fecha"] = pd.to_datetime(x_profiles["fecha"]).values
    return {
        "pca": pca,
        "features": feature_columns,
        "scores": scores_df,
        "explained": explained,
    }


def summarize_pca_loadings(pca_result: dict[str, object], component: int = 1) -> pd.DataFrame:
    """Summarize absolute PCA loadings by province for one component."""
    pca = pca_result["pca"]
    features = pca_result["features"]
    loadings = pd.Series(pca.components_[component - 1], index=features)
    rows = []
    for province in PROVINCE_ORDER:
        province_loadings = loadings[[f"{province}_{hour:02d}" for hour in HOURS]]
        rows.append(
            {
                "provincia": province,
                "loading_medio": province_loadings.mean(),
                "loading_abs_medio": province_loadings.abs().mean(),
                "hora_mayor_abs": int(province_loadings.abs().idxmax().split("_")[-1]),
            }
        )
    return pd.DataFrame(rows).sort_values("loading_abs_medio", ascending=False)


def plot_pca_explained(explained: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot individual and cumulative explained variance."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(1, len(explained) + 1)
    ax.bar(x, explained["varianza_explicada"], color="#4c78a8", label="Individual")
    ax.plot(x, explained["varianza_acumulada"], color="#f58518", marker="o", label="Acumulada")
    ax.set_title("PCA provincial: varianza explicada")
    ax.set_xlabel("Componente")
    ax.set_ylabel("Proporcion de varianza")
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    return ax


def plot_pca_scores_by_month(
    pca_result: dict[str, object],
    ax: Axes | None = None,
) -> Axes:
    """Plot PC1 vs PC2 colored by calendar month."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    scores = pca_result["scores"].copy()
    scores["mes"] = scores["fecha"].dt.month
    scatter = ax.scatter(scores["PC1"], scores["PC2"], c=scores["mes"], cmap="viridis", s=18)
    ax.set_title("Dias en espacio PCA provincial")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    colorbar = ax.figure.colorbar(scatter, ax=ax)
    colorbar.set_label("Mes")
    ax.grid(alpha=0.25)
    return ax


def assign_temporal_split(date: pd.Timestamp) -> str:
    """Assign a date to the temporal split defined for the provincial pipeline."""
    year = pd.Timestamp(date).year
    if year in [2019, 2021, 2022, 2023]:
        return "train"
    if year == 2024:
        return "validation"
    if year in [2025, 2026]:
        return "test"
    return "excluded"


def build_split_diagnostics(
    hourly_demand: pd.DataFrame,
    climate_daily: pd.DataFrame,
    pca_result: dict[str, object],
) -> pd.DataFrame:
    """Build daily diagnostics used to compare train, validation and test splits."""
    daily_total = hourly_demand.assign(
        demanda_total_mwh=hourly_demand[PROVINCE_ORDER].sum(axis=1)
    ).groupby("fecha")["demanda_total_mwh"].sum().reset_index()
    climate = climate_daily.copy()
    climate["fecha"] = pd.to_datetime(climate["fecha"])
    daily_temperature = climate.groupby("fecha")["temp_media"].mean().rename("temp_media_pais")
    shares = compute_hourly_shares(hourly_demand)
    daily_shares = shares.groupby("fecha")[["buenos_aires", "cordoba", "santa_fe"]].mean()
    daily_shares["share_top3_pct"] = daily_shares.sum(axis=1)
    diagnostics = daily_total.merge(daily_temperature, on="fecha", how="left")
    diagnostics = diagnostics.merge(daily_shares.reset_index(), on="fecha", how="left")
    diagnostics = diagnostics.merge(pca_result["scores"][["fecha", "PC1", "PC2"]], on="fecha", how="left")
    diagnostics["split"] = diagnostics["fecha"].apply(assign_temporal_split)
    return diagnostics.loc[diagnostics["split"] != "excluded"].copy()


def plot_split_diagnostics(diagnostics: pd.DataFrame) -> plt.Figure:
    """Plot compact boxplots comparing core variables across temporal splits."""
    split_order = ["train", "validation", "test"]
    specs = [
        ("demanda_total_mwh", "Demanda diaria total", "MWh/dia"),
        ("temp_media_pais", "Temperatura media provincial", "C"),
        ("PC1", "PCA PC1", "Score"),
        ("PC2", "PCA PC2", "Score"),
        ("buenos_aires", "Share Buenos Aires", "%"),
        ("share_top3_pct", "Share Buenos Aires + Cordoba + Santa Fe", "%"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (column, title, ylabel) in zip(axes.flatten(), specs):
        values = [
            diagnostics.loc[diagnostics["split"] == split, column].dropna()
            for split in split_order
        ]
        ax.boxplot(values, labels=split_order, showfliers=False)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Comparacion de distribuciones por split temporal", y=1.02)
    fig.tight_layout()
    return fig


def summarize_train_group_counts(hourly_demand: pd.DataFrame) -> pd.DataFrame:
    """Count train days available for each historical mean and KDE condition group."""
    daily = hourly_demand[["fecha", "mes", "dia_semana", "es_feriado"]].drop_duplicates()
    daily["split"] = daily["fecha"].apply(assign_temporal_split)
    train = daily.loc[daily["split"] == "train"]
    full_index = pd.MultiIndex.from_product(
        [range(1, 13), range(7), [0, 1]],
        names=["mes", "dia_semana", "es_feriado"],
    )
    counts = train.groupby(["mes", "dia_semana", "es_feriado"]).size()
    result = counts.reindex(full_index, fill_value=0).rename("dias_train").reset_index()
    return result.sort_values("dias_train")


def summarize_group_count_risk(group_counts: pd.DataFrame) -> pd.DataFrame:
    """Summarize how many condition groups have low train support."""
    thresholds = [0, 1, 2, 3, 5, 10]
    rows = []
    for threshold in thresholds:
        rows.append(
            {
                "umbral": f"<= {threshold}",
                "grupos": int((group_counts["dias_train"] <= threshold).sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_train_group_count_histogram(group_counts: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot the distribution of train sample counts per condition group."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.hist(group_counts["dias_train"], bins=range(0, int(group_counts["dias_train"].max()) + 2), color="#4c78a8")
    ax.set_title("Dias de train por grupo (mes, dia_semana, es_feriado)")
    ax.set_xlabel("Dias disponibles en train")
    ax.set_ylabel("Cantidad de grupos")
    ax.grid(axis="y", alpha=0.25)
    return ax


def plot_pca_loading_heatmap(
    pca_result: dict[str, object],
    component: int,
    ax: Axes | None = None,
) -> Axes:
    """Plot PCA loadings as a province-by-hour heatmap for one component."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    pca = pca_result["pca"]
    features = pca_result["features"]
    loadings = pd.Series(pca.components_[component - 1], index=features)
    matrix = pd.DataFrame(
        [
            [loadings[f"{province}_{hour:02d}"] for hour in HOURS]
            for province in PROVINCE_ORDER
        ],
        index=PROVINCE_ORDER,
        columns=HOURS,
    )
    limit = float(np.abs(matrix.to_numpy()).max())
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_title(f"Loadings PCA — PC{component}")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Provincia")
    ax.set_xticks(range(len(HOURS)))
    ax.set_xticklabels(HOURS)
    ax.set_yticks(range(len(PROVINCE_ORDER)))
    ax.set_yticklabels(PROVINCE_ORDER)
    colorbar = ax.figure.colorbar(image, ax=ax)
    colorbar.set_label("Loading")
    return ax


def plot_calendar_profiles_selected(
    hourly_demand: pd.DataFrame,
    provinces: list[str],
) -> plt.Figure:
    """Plot normalized calendar profiles for selected provinces."""
    data = hourly_demand.copy()
    data["tipo_semana"] = np.where(data["dia_semana"].isin([5, 6]), "fin_de_semana", "habil")
    fig, axes = plt.subplots(len(provinces), 2, figsize=(11, 2.4 * len(provinces)), sharex=True)
    for row, province in enumerate(provinces):
        for label, group in data.groupby("tipo_semana"):
            profile = group.groupby("hora")[province].mean()
            axes[row, 0].plot(profile.index, profile / profile.mean(), label=label)
        for label, group in data.groupby("es_feriado"):
            profile = group.groupby("hora")[province].mean()
            axes[row, 1].plot(
                profile.index,
                profile / profile.mean(),
                label="feriado" if label == 1 else "no_feriado",
            )
        axes[row, 0].set_ylabel(province)
        axes[row, 0].grid(alpha=0.2)
        axes[row, 1].grid(alpha=0.2)
    axes[0, 0].set_title("Habil vs fin de semana")
    axes[0, 1].set_title("Feriado vs no feriado")
    axes[-1, 0].set_xlabel("Hora")
    axes[-1, 1].set_xlabel("Hora")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("Efecto calendario en perfiles provinciales seleccionados", y=1.01)
    fig.tight_layout()
    return fig
