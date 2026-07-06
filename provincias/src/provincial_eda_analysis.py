from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
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
