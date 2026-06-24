import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


DEMAND_COLUMNS = [f"demanda_{hour:02d}" for hour in range(1, 25)]
HOURS = list(range(1, 25))

OUTLIER_METRICS = [
    "energia_diaria_mwh",
    "demanda_maxima_mwh",
    "demanda_minima_mwh",
    "amplitud_diaria_mwh",
    "max_rampa_abs_mwh",
    "shape_pca_distance",
]

METRIC_LABELS = {
    "energia_diaria_mwh": "energia diaria",
    "demanda_maxima_mwh": "demanda maxima",
    "demanda_minima_mwh": "demanda minima",
    "amplitud_diaria_mwh": "amplitud diaria",
    "max_rampa_abs_mwh": "rampa horaria",
    "shape_pca_distance": "forma del perfil",
}

SPECIAL_CALENDAR_DATES = {
    "01-01": "Anio Nuevo",
    "05-01": "Dia del trabajador",
    "07-09": "Dia de la Independencia",
    "12-24": "Nochebuena",
    "12-25": "Navidad",
    "12-31": "Fin de anio",
}

EVENT_DATES = {
    "2022-11-22": "Mundial Qatar 2022",
    "2022-11-26": "Mundial Qatar 2022",
    "2022-11-30": "Mundial Qatar 2022",
    "2022-12-03": "Mundial Qatar 2022",
    "2022-12-09": "Mundial Qatar 2022",
    "2022-12-13": "Mundial Qatar 2022",
    "2022-12-18": "Mundial Qatar 2022",
    "2022-12-20": "festejos seleccion",
    "2023-08-13": "elecciones 2023",
    "2023-10-22": "elecciones 2023",
    "2023-11-19": "elecciones 2023",
    "2023-03-01": "evento de suministro",
}


def _robust_zscore(values: pd.Series) -> pd.Series:
    """Compute robust z-scores using the median absolute deviation."""
    median = values.median()
    mad = (values - median).abs().median()
    if mad == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return 0.6745 * (values - median) / mad


def _assign_day_type(row: pd.Series) -> str:
    """Assign day type when the input table does not already include it."""
    if row["es_feriado"] == 1:
        return "feriado"
    if row["dia_semana"] == 5:
        return "sabado"
    if row["dia_semana"] == 6:
        return "domingo"
    return "habil"


def _build_explanation(row: pd.Series, thresholds: dict[str, float]) -> str:
    """Classify a potentially atypical day using calendar, climate and events."""
    explanations = []
    date_key = row["fecha"].date().isoformat()
    month_day = row["fecha"].strftime("%m-%d")

    if date_key in EVENT_DATES:
        explanations.append(EVENT_DATES[date_key])
    if row["es_feriado"] == 1:
        explanations.append("feriado")
    if month_day in SPECIAL_CALENDAR_DATES:
        explanations.append(SPECIAL_CALENDAR_DATES[month_day])
    if row["temp_media"] >= thresholds["temp_media_p90"] or row["temp_max"] >= thresholds["temp_max_p90"]:
        explanations.append("calor extremo")
    if row["temp_media"] <= thresholds["temp_media_p10"] or row["grado_frio"] >= thresholds["grado_frio_p90"]:
        explanations.append("frio extremo")
    if (
        abs(row.get("rz_shape_pca_distance", 0)) > 3.5
        and row["hora_pico"] not in [16, 21, 22]
    ):
        explanations.append("forma horaria atipica")

    if not explanations:
        return "sin explicacion clara"

    return " + ".join(dict.fromkeys(explanations))


def _recommend_action(row: pd.Series) -> str:
    """Suggest whether the day should be learned as real variability or flagged."""
    explanation = row["explicacion"]
    if "evento de suministro" in explanation:
        return "etiquetar aparte / revisar"
    if "forma horaria atipica" in explanation:
        return "revisar forma / posible etiqueta"
    if explanation == "sin explicacion clara":
        return "revisar antes de modelar"
    return "conservar como variabilidad real"


def build_outlier_features(
    daily_demand: pd.DataFrame,
    conditions: pd.DataFrame,
) -> pd.DataFrame:
    """Build daily features and robust scores used to detect atypical days."""
    climate_columns = [
        "fecha",
        "temp_media",
        "temp_max",
        "grado_calor",
        "grado_frio",
    ]
    missing_climate = [
        column for column in climate_columns if column != "fecha" and column not in daily_demand.columns
    ]
    if missing_climate:
        df = daily_demand.merge(
            conditions[climate_columns],
            on="fecha",
            how="left",
            suffixes=("", "_cond"),
        )
    else:
        df = daily_demand.copy()

    if "tipo_dia" not in df.columns:
        df["tipo_dia"] = df.apply(_assign_day_type, axis=1)

    ramps = df[DEMAND_COLUMNS].diff(axis=1)
    ramps.iloc[:, 0] = np.nan
    df["max_rampa_subida_mwh"] = ramps.max(axis=1)
    df["max_rampa_bajada_mwh"] = ramps.min(axis=1)
    df["max_rampa_abs_mwh"] = ramps.abs().max(axis=1)
    df["hora_max_rampa_abs"] = ramps.abs().idxmax(axis=1).str[-2:].astype(int)

    normalized_profiles = df[DEMAND_COLUMNS].div(df["demanda_media_mwh"], axis=0)
    pca = PCA(n_components=6)
    scores = pca.fit_transform(normalized_profiles)
    score_scale = scores.std(axis=0, ddof=0)
    df["shape_pca_distance"] = np.sqrt(((scores / score_scale) ** 2).sum(axis=1))
    df.attrs["shape_pca_explained_variance"] = pca.explained_variance_ratio_.tolist()

    for metric in OUTLIER_METRICS:
        df[f"rz_{metric}"] = _robust_zscore(df[metric])

    robust_columns = [f"rz_{metric}" for metric in OUTLIER_METRICS]
    abs_scores = df[robust_columns].abs()
    df["outlier_score"] = abs_scores.max(axis=1)
    df["metric_principal"] = (
        abs_scores.idxmax(axis=1).str.replace("rz_", "", regex=False).map(METRIC_LABELS)
    )

    thresholds = {
        "temp_media_p10": df["temp_media"].quantile(0.10),
        "temp_media_p90": df["temp_media"].quantile(0.90),
        "temp_max_p90": df["temp_max"].quantile(0.90),
        "grado_frio_p90": df["grado_frio"].quantile(0.90),
    }
    df["explicacion"] = df.apply(_build_explanation, axis=1, thresholds=thresholds)
    df["accion_modelado"] = df.apply(_recommend_action, axis=1)
    return df


def summarize_outlier_counts(
    outlier_features: pd.DataFrame,
    threshold: float = 3.5,
) -> pd.DataFrame:
    """Count robust outliers by metric."""
    rows = []
    for metric in OUTLIER_METRICS:
        score = outlier_features[f"rz_{metric}"]
        most_extreme_idx = score.abs().idxmax()
        rows.append(
            {
                "metrica": METRIC_LABELS[metric],
                "outliers_abs": int(score.abs().gt(threshold).sum()),
                "outliers_altos": int(score.gt(threshold).sum()),
                "outliers_bajos": int(score.lt(-threshold).sum()),
                "score_max_abs": score.abs().max(),
                "fecha_mas_extrema": outlier_features.loc[most_extreme_idx, "fecha"].date().isoformat(),
            }
        )

    return pd.DataFrame(rows).round({"score_max_abs": 2})


def summarize_extreme_days_by_metric(outlier_features: pd.DataFrame) -> pd.DataFrame:
    """Return representative extreme days for the main outlier dimensions."""
    metric_specs = [
        ("energia minima", "energia_diaria_mwh", "min"),
        ("energia maxima", "energia_diaria_mwh", "max"),
        ("demanda maxima", "demanda_maxima_mwh", "max"),
        ("demanda minima", "demanda_minima_mwh", "min"),
        ("amplitud maxima", "amplitud_diaria_mwh", "max"),
        ("rampa maxima", "max_rampa_abs_mwh", "max"),
        ("forma mas atipica", "shape_pca_distance", "max"),
    ]
    rows = []

    for label, metric, direction in metric_specs:
        idx = outlier_features[metric].idxmin() if direction == "min" else outlier_features[metric].idxmax()
        row = outlier_features.loc[idx]
        rows.append(
            {
                "caso": label,
                "fecha": row["fecha"].date().isoformat(),
                "tipo_dia": row["tipo_dia"],
                "temp_media_c": row["temp_media"],
                "energia_gwh": row["energia_diaria_mwh"] / 1000,
                "hora_pico": int(row["hora_pico"]),
                "valor": row[metric],
                "score_robusto": row[f"rz_{metric}"] if f"rz_{metric}" in row else np.nan,
                "explicacion": row["explicacion"],
                "accion_modelado": row["accion_modelado"],
            }
        )

    return pd.DataFrame(rows).round(
        {
            "temp_media_c": 2,
            "energia_gwh": 2,
            "valor": 2,
            "score_robusto": 2,
        }
    )


def summarize_top_outlier_days(
    outlier_features: pd.DataFrame,
    top_n: int = 12,
) -> pd.DataFrame:
    """Return the days with the highest overall outlier score."""
    columns = [
        "fecha",
        "tipo_dia",
        "temp_media",
        "temp_max",
        "energia_diaria_mwh",
        "hora_pico",
        "max_rampa_abs_mwh",
        "metric_principal",
        "outlier_score",
        "explicacion",
        "accion_modelado",
    ]
    result = outlier_features.nlargest(top_n, "outlier_score")[columns].copy()
    result["fecha"] = result["fecha"].dt.date.astype(str)
    result["energia_gwh"] = result["energia_diaria_mwh"] / 1000
    result = result.drop(columns=["energia_diaria_mwh"])
    return result[
        [
            "fecha",
            "tipo_dia",
            "temp_media",
            "temp_max",
            "energia_gwh",
            "hora_pico",
            "max_rampa_abs_mwh",
            "metric_principal",
            "outlier_score",
            "explicacion",
            "accion_modelado",
        ]
    ].round(
        {
            "temp_media": 2,
            "temp_max": 2,
            "energia_gwh": 2,
            "max_rampa_abs_mwh": 2,
            "outlier_score": 2,
        }
    )


def summarize_peak_hour_outliers(outlier_features: pd.DataFrame) -> pd.DataFrame:
    """Summarize rare daily peak hours."""
    distribution = (
        outlier_features["hora_pico"]
        .value_counts()
        .sort_index()
        .rename_axis("hora_pico")
        .reset_index(name="dias")
    )
    distribution["porcentaje"] = distribution["dias"] / len(outlier_features) * 100
    distribution["categoria"] = np.where(
        distribution["hora_pico"].isin([16, 21, 22]),
        "habitual",
        "menos frecuente",
    )
    return distribution.round({"porcentaje": 1})


def plot_outlier_score_timeline(outlier_features: pd.DataFrame) -> None:
    """Plot the maximum robust outlier score through time."""
    df = outlier_features.sort_values("fecha")
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(df["fecha"], df["outlier_score"], color="#9ecae1", linewidth=0.9)
    ax.scatter(df["fecha"], df["outlier_score"], color="#4878CF", s=12, alpha=0.55)
    ax.axhline(3.5, color="#D65F5F", linewidth=1.2, linestyle="--", label="umbral 3.5")

    top_rows = df.nlargest(5, "outlier_score")
    for _, row in top_rows.iterrows():
        ax.annotate(
            row["fecha"].strftime("%Y-%m-%d"),
            (row["fecha"], row["outlier_score"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    ax.set_title("Score maximo de outlier por dia")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Score robusto maximo")
    ax.grid(linewidth=0.4, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_ramp_outlier_detail(
    outlier_features: pd.DataFrame,
    date: str = "2023-03-01",
) -> None:
    """Plot the hourly profile for a day with an extreme ramp."""
    row = outlier_features.loc[outlier_features["fecha"].eq(pd.Timestamp(date))].iloc[0]
    values = row[DEMAND_COLUMNS].astype(float)
    ramp_hour = int(row["hora_max_rampa_abs"])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(HOURS, values / 1000, marker="o", color="#D65F5F", linewidth=2)
    ax.axvspan(ramp_hour - 1, ramp_hour, color="#F4B400", alpha=0.25, label="rampa extrema")
    ax.set_title(f"Detalle horario del outlier de rampa ({date})")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Demanda (GWh)")
    ax.set_xticks(HOURS)
    ax.grid(linewidth=0.4, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_top_outlier_profiles(
    outlier_features: pd.DataFrame,
    top_n: int = 6,
) -> None:
    """Plot the profiles of the highest-scoring outlier days."""
    top_rows = outlier_features.nlargest(top_n, "outlier_score")
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=False)

    for ax, (_, row) in zip(axes.flatten(), top_rows.iterrows()):
        ax.plot(HOURS, row[DEMAND_COLUMNS].astype(float) / 1000, color="#4878CF", linewidth=1.8)
        ax.set_title(
            f"{row['fecha'].date()} | {row['metric_principal']}",
            fontsize=9,
        )
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.grid(linewidth=0.4, alpha=0.5)

    for ax in axes[:, 0]:
        ax.set_ylabel("Demanda (GWh)")
    for ax in axes[1, :]:
        ax.set_xlabel("Hora")

    fig.suptitle("Perfiles de los dias mas atipicos", y=1.02)
    fig.tight_layout()
    plt.show()
