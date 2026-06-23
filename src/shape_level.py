import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.decomposition import PCA

from src.daily_demand import DAY_TYPE_ORDER, DEMAND_COLUMNS, HOURS


def build_normalized_profiles(daily_demand: pd.DataFrame) -> pd.DataFrame:
    """Normalize each daily profile by its own mean demand."""
    normalized_profiles = daily_demand[DEMAND_COLUMNS].div(
        daily_demand["demanda_media_mwh"],
        axis=0,
    )
    normalized_profiles.columns = DEMAND_COLUMNS
    return normalized_profiles


def fit_pca(data: pd.DataFrame, n_components: int = 4) -> tuple[PCA, np.ndarray]:
    """Fit PCA to a profile matrix and return the model and scores."""
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(data)
    return pca, scores


def summarize_shape_level(daily_demand: pd.DataFrame) -> pd.DataFrame:
    """Summarize how much variation is associated with level and shape."""
    raw_profiles = daily_demand[DEMAND_COLUMNS]
    normalized_profiles = build_normalized_profiles(daily_demand)
    daily_level = daily_demand["demanda_media_mwh"].to_numpy()
    raw_values = raw_profiles.to_numpy()
    total_variability = ((raw_values - raw_values.mean()) ** 2).sum()
    level_residual = ((raw_values - daily_level[:, None]) ** 2).sum()
    level_explained = 1 - level_residual / total_variability
    raw_pca, raw_scores = fit_pca(raw_profiles)
    shape_pca, shape_scores = fit_pca(normalized_profiles)
    raw_pc1_energy_corr = abs(
        np.corrcoef(raw_scores[:, 0], daily_demand["energia_diaria_mwh"])[0, 1]
    )
    shape_pc1_energy_corr = abs(
        np.corrcoef(shape_scores[:, 0], daily_demand["energia_diaria_mwh"])[0, 1]
    )
    relative_amplitude = (
        daily_demand["amplitud_diaria_mwh"] / daily_demand["demanda_media_mwh"]
    )
    rows = [
        {
            "metric": "cv_energia_diaria",
            "value": daily_demand["energia_diaria_mwh"].std()
            / daily_demand["energia_diaria_mwh"].mean(),
        },
        {
            "metric": "amplitud_relativa_media",
            "value": relative_amplitude.mean(),
        },
        {
            "metric": "variabilidad_explicada_por_nivel",
            "value": level_explained,
        },
        {
            "metric": "pca_crudo_pc1_varianza",
            "value": raw_pca.explained_variance_ratio_[0],
        },
        {
            "metric": "pca_crudo_pc1_corr_energia_abs",
            "value": raw_pc1_energy_corr,
        },
        {
            "metric": "pca_forma_pc1_varianza",
            "value": shape_pca.explained_variance_ratio_[0],
        },
        {
            "metric": "pca_forma_pc2_varianza",
            "value": shape_pca.explained_variance_ratio_[1],
        },
        {
            "metric": "pca_forma_pc1_corr_energia_abs",
            "value": shape_pc1_energy_corr,
        },
    ]
    return pd.DataFrame(rows).round({"value": 4})


def get_pca_summary(daily_demand: pd.DataFrame) -> pd.DataFrame:
    """Return explained variance and energy correlation for raw and normalized PCA."""
    raw_profiles = daily_demand[DEMAND_COLUMNS]
    normalized_profiles = build_normalized_profiles(daily_demand)
    raw_pca, raw_scores = fit_pca(raw_profiles)
    shape_pca, shape_scores = fit_pca(normalized_profiles)
    rows = []
    for component in range(4):
        rows.append(
            {
                "dataset": "demanda_cruda",
                "component": component + 1,
                "explained_variance": raw_pca.explained_variance_ratio_[component],
                "abs_corr_energia": abs(
                    np.corrcoef(
                        raw_scores[:, component],
                        daily_demand["energia_diaria_mwh"],
                    )[0, 1]
                ),
            }
        )
        rows.append(
            {
                "dataset": "perfil_normalizado",
                "component": component + 1,
                "explained_variance": shape_pca.explained_variance_ratio_[component],
                "abs_corr_energia": abs(
                    np.corrcoef(
                        shape_scores[:, component],
                        daily_demand["energia_diaria_mwh"],
                    )[0, 1]
                ),
            }
        )
    return pd.DataFrame(rows).round(4)


def plot_energy_examples(
    daily_demand: pd.DataFrame,
    ax: Axes | None = None,
) -> Axes:
    """Plot low, median, and high daily energy profiles in raw MWh."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    ordered = daily_demand.sort_values("energia_diaria_mwh")
    selected_rows = [
        ordered.iloc[0],
        ordered.iloc[len(ordered) // 2],
        ordered.iloc[-1],
    ]
    labels = ["energia baja", "energia mediana", "energia alta"]
    for label, row in zip(labels, selected_rows):
        ax.plot(
            HOURS,
            row[DEMAND_COLUMNS],
            linewidth=2,
            label=f"{label} ({row['fecha'].date()})",
        )
    ax.set_title("Perfiles diarios con distinto nivel de consumo")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Demanda (MWh)")
    ax.set_xticks(HOURS)
    ax.grid(alpha=0.25)
    ax.legend()
    return ax


def plot_normalized_profile_by_day_type(
    daily_demand: pd.DataFrame,
    ax: Axes | None = None,
) -> Axes:
    """Plot average normalized profile by day type."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    normalized_profiles = build_normalized_profiles(daily_demand)
    plot_data = pd.concat(
        [daily_demand[["tipo_dia"]].reset_index(drop=True), normalized_profiles],
        axis=1,
    )
    for day_type in DAY_TYPE_ORDER:
        subset = plot_data[plot_data["tipo_dia"] == day_type]
        if not subset.empty:
            ax.plot(HOURS, subset[DEMAND_COLUMNS].mean(), linewidth=2, label=day_type)
    ax.axhline(1, color="black", linewidth=1, alpha=0.4)
    ax.set_title("Forma promedio normalizada por tipo de dia")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Demanda relativa a la media diaria")
    ax.set_xticks(HOURS)
    ax.grid(alpha=0.25)
    ax.legend(title="Tipo de dia")
    return ax


def plot_raw_pc1_vs_energy(
    daily_demand: pd.DataFrame,
    ax: Axes | None = None,
) -> Axes:
    """Plot the first raw PCA score against daily energy."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    raw_pca, raw_scores = fit_pca(daily_demand[DEMAND_COLUMNS])
    ax.scatter(
        daily_demand["energia_diaria_mwh"] / 1000,
        raw_scores[:, 0],
        s=16,
        alpha=0.55,
        color="#4c78a8",
    )
    corr = np.corrcoef(raw_scores[:, 0], daily_demand["energia_diaria_mwh"])[0, 1]
    ax.set_title(f"PC1 crudo vs energia diaria (corr={corr:.3f})")
    ax.set_xlabel("Energia diaria (GWh)")
    ax.set_ylabel("Score PC1")
    ax.grid(alpha=0.25)
    return ax


def plot_shape_pca_components(
    daily_demand: pd.DataFrame,
    ax: Axes | None = None,
) -> Axes:
    """Plot the first two PCA components fitted to normalized profiles."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    normalized_profiles = build_normalized_profiles(daily_demand)
    shape_pca, _ = fit_pca(normalized_profiles, n_components=2)
    ax.plot(
        HOURS,
        shape_pca.components_[0],
        linewidth=2,
        label=f"PC1 ({shape_pca.explained_variance_ratio_[0] * 100:.1f}%)",
    )
    ax.plot(
        HOURS,
        shape_pca.components_[1],
        linewidth=2,
        label=f"PC2 ({shape_pca.explained_variance_ratio_[1] * 100:.1f}%)",
    )
    ax.axhline(0, color="black", linewidth=1, alpha=0.4)
    ax.set_title("Componentes principales de la forma normalizada")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Loading")
    ax.set_xticks(HOURS)
    ax.grid(alpha=0.25)
    ax.legend()
    return ax
