import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEMAND_COLUMNS = [f"demanda_{hour:02d}" for hour in range(1, 25)]
HOURS = list(range(1, 25))


WORLD_CUP_EVENTS = [
    {
        "evento": "Arabia Saudita",
        "fecha": "2022-11-22",
        "hora_evento": "07-09",
        "horas": [7, 8, 9],
        "same_holiday": True,
    },
    {
        "evento": "Mexico",
        "fecha": "2022-11-26",
        "hora_evento": "16-18",
        "horas": [16, 17, 18],
        "same_holiday": True,
    },
    {
        "evento": "Polonia",
        "fecha": "2022-11-30",
        "hora_evento": "16-18",
        "horas": [16, 17, 18],
        "same_holiday": True,
    },
    {
        "evento": "Australia",
        "fecha": "2022-12-03",
        "hora_evento": "16-18",
        "horas": [16, 17, 18],
        "same_holiday": True,
    },
    {
        "evento": "Paises Bajos",
        "fecha": "2022-12-09",
        "hora_evento": "16-19",
        "horas": [16, 17, 18, 19],
        "same_holiday": True,
    },
    {
        "evento": "Croacia",
        "fecha": "2022-12-13",
        "hora_evento": "16-18",
        "horas": [16, 17, 18],
        "same_holiday": True,
    },
    {
        "evento": "Francia final",
        "fecha": "2022-12-18",
        "hora_evento": "12-15",
        "horas": [12, 13, 14, 15],
        "same_holiday": True,
    },
    {
        "evento": "Festejos seleccion",
        "fecha": "2022-12-20",
        "hora_evento": "dia completo",
        "horas": None,
        "same_holiday": False,
    },
]


ELECTION_EVENTS = [
    {"evento": "PASO", "fecha": "2023-08-13"},
    {"evento": "General", "fecha": "2023-10-22"},
    {"evento": "Balotaje", "fecha": "2023-11-19"},
]


COVID_PERIODS = [
    {
        "periodo": "Abr-may 2021",
        "inicio": "2021-04-01",
        "fin": "2021-05-31",
    },
    {
        "periodo": "Restriccion 22-31 may 2021",
        "inicio": "2021-05-22",
        "fin": "2021-05-31",
    },
    {
        "periodo": "Enero 2022 Omicron",
        "inicio": "2022-01-01",
        "fin": "2022-01-31",
    },
]


def _prepare_event_frame(
    daily_demand: pd.DataFrame,
    conditions: pd.DataFrame,
) -> pd.DataFrame:
    """Merge daily demand features with climate columns needed for controls."""
    climate_columns = [
        "fecha",
        "temp_media",
        "temp_max",
        "grado_calor",
        "grado_frio",
    ]
    df = daily_demand.merge(
        conditions[climate_columns],
        on="fecha",
        how="left",
        suffixes=("", "_cond"),
    )
    df["anio"] = df["fecha"].dt.year
    df["grupo_semana"] = np.where(df["dia_semana"] < 5, "habil", "fin_de_semana")
    return df


def _control_subset(
    df: pd.DataFrame,
    event_date: str | pd.Timestamp,
    temp_window: float = 3,
    min_controls: int = 5,
    same_holiday: bool = True,
) -> tuple[pd.Series, pd.DataFrame, str]:
    """
    Find comparable days for one event.

    The strict comparison uses same month, weekday, holiday flag and a similar
    mean temperature. If that leaves too few controls, the function relaxes the
    holiday flag first because some social events created exceptional holidays.
    """
    event_date = pd.Timestamp(event_date)
    event_row = df.loc[df["fecha"].eq(event_date)].iloc[0]

    control_specs = []
    if same_holiday:
        control_specs.append(
            {
                "same_weekday": True,
                "same_holiday": True,
                "same_temperature": True,
                "label": "mismo mes, dia, feriado y temperatura similar",
            }
        )

    control_specs.extend(
        [
            {
                "same_weekday": True,
                "same_holiday": False,
                "same_temperature": True,
                "label": "mismo mes, dia y temperatura similar",
            },
            {
                "same_weekday": True,
                "same_holiday": False,
                "same_temperature": False,
                "label": "mismo mes y dia",
            },
            {
                "same_weekday": False,
                "same_holiday": False,
                "same_temperature": True,
                "label": "mismo mes y temperatura similar",
            },
            {
                "same_weekday": False,
                "same_holiday": False,
                "same_temperature": False,
                "label": "mismo mes",
            },
        ]
    )

    for spec in control_specs:
        mask = df["fecha"].ne(event_date) & df["mes"].eq(event_row["mes"])
        if spec["same_weekday"]:
            mask &= df["dia_semana"].eq(event_row["dia_semana"])
        if spec["same_holiday"]:
            mask &= df["es_feriado"].eq(event_row["es_feriado"])
        if spec["same_temperature"]:
            mask &= df["temp_media"].sub(event_row["temp_media"]).abs().le(temp_window)

        controls = df.loc[mask].copy()
        if len(controls) >= min_controls:
            return event_row, controls, spec["label"]

    return event_row, controls, control_specs[-1]["label"]


def _percentage_difference(observed: float, expected: float) -> float:
    """Return percentage difference between observed and expected values."""
    if pd.isna(expected) or expected == 0:
        return np.nan
    return (observed / expected - 1) * 100


def _hour_block_difference(
    event_row: pd.Series,
    controls: pd.DataFrame,
    hours: list[int] | None,
) -> tuple[float, float, float]:
    """Compare the mean demand of a selected hour block against controls."""
    if not hours:
        return np.nan, np.nan, np.nan

    hour_columns = [f"demanda_{hour:02d}" for hour in hours]
    observed = event_row[hour_columns].mean()
    expected = controls[hour_columns].mean(axis=1).mean()
    return observed, expected, _percentage_difference(observed, expected)


def summarize_covid_periods(
    daily_demand: pd.DataFrame,
    conditions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize selected COVID-related periods against comparable days.

    Residuals are computed day by day against same-month comparable controls.
    """
    df = _prepare_event_frame(daily_demand, conditions)
    rows = []

    for period in COVID_PERIODS:
        period_mask = df["fecha"].between(period["inicio"], period["fin"])
        for group_label, group_mask in [
            ("todos", period_mask),
            ("habil", period_mask & df["grupo_semana"].eq("habil")),
            ("fin_de_semana", period_mask & df["grupo_semana"].eq("fin_de_semana")),
        ]:
            subset = df.loc[group_mask]
            residuals = []
            for event_date in subset["fecha"]:
                event_row, controls, _ = _control_subset(df, event_date)
                expected = controls["energia_diaria_mwh"].mean()
                residuals.append(_percentage_difference(event_row["energia_diaria_mwh"], expected))

            rows.append(
                {
                    "periodo": period["periodo"],
                    "grupo": group_label,
                    "dias": len(subset),
                    "energia_media_gwh": subset["energia_diaria_mwh"].mean() / 1000,
                    "residuo_medio_pct": np.nanmean(residuals),
                    "residuo_mediano_pct": np.nanmedian(residuals),
                }
            )

    return pd.DataFrame(rows).round(2)


def summarize_annual_normalization(
    daily_demand: pd.DataFrame,
    conditions: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize annual demand and comparable-day residuals by weekday group."""
    df = _prepare_event_frame(daily_demand, conditions)
    rows = []

    for _, event_row in df.iterrows():
        _, controls, _ = _control_subset(df, event_row["fecha"])
        expected = controls["energia_diaria_mwh"].mean()
        rows.append(
            {
                "anio": event_row["anio"],
                "grupo": event_row["grupo_semana"],
                "energia_gwh": event_row["energia_diaria_mwh"] / 1000,
                "residuo_pct": _percentage_difference(event_row["energia_diaria_mwh"], expected),
            }
        )

    residuals = pd.DataFrame(rows)
    summary = (
        residuals.groupby(["anio", "grupo"])
        .agg(
            energia_media_gwh=("energia_gwh", "mean"),
            residuo_medio_pct=("residuo_pct", "mean"),
            residuo_mediano_pct=("residuo_pct", "median"),
            dias=("residuo_pct", "count"),
        )
        .reset_index()
    )
    return summary.round(2)


def summarize_world_cup_events(
    daily_demand: pd.DataFrame,
    conditions: pd.DataFrame,
) -> pd.DataFrame:
    """Compare Argentina World Cup dates against comparable control days."""
    df = _prepare_event_frame(daily_demand, conditions)
    return _summarize_event_collection(df, WORLD_CUP_EVENTS)


def summarize_election_events(
    daily_demand: pd.DataFrame,
    conditions: pd.DataFrame,
) -> pd.DataFrame:
    """Compare 2023 election Sundays against comparable Sundays."""
    df = _prepare_event_frame(daily_demand, conditions)
    rows = []

    for event in ELECTION_EVENTS:
        event_row, controls, control_label = _control_subset(df, event["fecha"])
        morning = _hour_block_difference(event_row, controls, list(range(8, 13)))
        afternoon = _hour_block_difference(event_row, controls, list(range(13, 19)))
        night = _hour_block_difference(event_row, controls, list(range(19, 24)))

        next_day = event_row["fecha"] + pd.Timedelta(days=1)
        next_row, next_controls, _ = _control_subset(df, next_day)
        next_expected = next_controls["energia_diaria_mwh"].mean()

        rows.append(
            {
                "evento": event["evento"],
                "fecha": event_row["fecha"].date().isoformat(),
                "n_controles": len(controls),
                "control": control_label,
                "temp_media_c": event_row["temp_media"],
                "energia_gwh": event_row["energia_diaria_mwh"] / 1000,
                "energia_esperada_gwh": controls["energia_diaria_mwh"].mean() / 1000,
                "dif_energia_pct": _percentage_difference(
                    event_row["energia_diaria_mwh"],
                    controls["energia_diaria_mwh"].mean(),
                ),
                "dif_manana_pct": morning[2],
                "dif_tarde_pct": afternoon[2],
                "dif_noche_pct": night[2],
                "dif_dia_posterior_pct": _percentage_difference(
                    next_row["energia_diaria_mwh"],
                    next_expected,
                ),
            }
        )

    return pd.DataFrame(rows).round(2)


def _summarize_event_collection(
    df: pd.DataFrame,
    events: list[dict],
) -> pd.DataFrame:
    """Summarize a list of day-level events using comparable controls."""
    rows = []

    for event in events:
        event_row, controls, control_label = _control_subset(
            df,
            event["fecha"],
            same_holiday=event.get("same_holiday", True),
        )
        block_observed, block_expected, block_difference = _hour_block_difference(
            event_row,
            controls,
            event.get("horas"),
        )
        expected_energy = controls["energia_diaria_mwh"].mean()

        rows.append(
            {
                "evento": event["evento"],
                "fecha": event_row["fecha"].date().isoformat(),
                "hora_evento": event.get("hora_evento", ""),
                "n_controles": len(controls),
                "control": control_label,
                "feriado": int(event_row["es_feriado"]),
                "temp_media_c": event_row["temp_media"],
                "energia_gwh": event_row["energia_diaria_mwh"] / 1000,
                "energia_esperada_gwh": expected_energy / 1000,
                "dif_energia_pct": _percentage_difference(event_row["energia_diaria_mwh"], expected_energy),
                "hora_pico": int(event_row["hora_pico"]),
                "demanda_evento_mwh": block_observed,
                "demanda_esperada_mwh": block_expected,
                "dif_bloque_pct": block_difference,
            }
        )

    return pd.DataFrame(rows).round(2)


def plot_covid_normalization(
    daily_demand: pd.DataFrame,
    conditions: pd.DataFrame,
) -> None:
    """Plot annual comparable-day residuals for weekdays and weekends."""
    summary = summarize_annual_normalization(daily_demand, conditions)
    fig, ax = plt.subplots(figsize=(8, 4))

    palette = {"habil": "#4878CF", "fin_de_semana": "#D65F5F"}
    for group, subset in summary.groupby("grupo"):
        ax.plot(
            subset["anio"],
            subset["residuo_medio_pct"],
            marker="o",
            linewidth=1.8,
            color=palette.get(group, "gray"),
            label=group,
        )

    ax.axhline(0, color="#1a1a2e", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_title("Residuo anual contra dias comparables")
    ax.set_xlabel("Anio")
    ax.set_ylabel("Residuo medio de energia (%)")
    ax.grid(linewidth=0.4, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_event_energy_residuals(summary: pd.DataFrame, title: str) -> None:
    """Bar plot of day-level residuals for a summary table."""
    fig, ax = plt.subplots(figsize=(9, 4))

    colors = np.where(summary["dif_energia_pct"] >= 0, "#4878CF", "#D65F5F")
    ax.bar(summary["evento"], summary["dif_energia_pct"], color=colors, alpha=0.8)
    ax.axhline(0, color="#1a1a2e", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel("Diferencia vs controles (%)")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    plt.show()


def plot_world_cup_profiles(
    daily_demand: pd.DataFrame,
    conditions: pd.DataFrame,
) -> None:
    """Plot observed and expected hourly profiles for Argentina match days."""
    df = _prepare_event_frame(daily_demand, conditions)
    match_events = [event for event in WORLD_CUP_EVENTS if event["horas"]]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for ax, event in zip(axes_flat, match_events):
        event_row, controls, _ = _control_subset(
            df,
            event["fecha"],
            same_holiday=event.get("same_holiday", True),
        )
        expected_profile = controls[DEMAND_COLUMNS].mean()
        observed_profile = event_row[DEMAND_COLUMNS]
        hours = event["horas"]

        ax.plot(HOURS, observed_profile / 1000, color="#D65F5F", linewidth=1.8, label="evento")
        ax.plot(HOURS, expected_profile / 1000, color="#4878CF", linewidth=1.8, label="control")
        ax.axvspan(min(hours) - 0.5, max(hours) + 0.5, color="#F4B400", alpha=0.18)
        ax.set_title(event["evento"], fontsize=10)
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.grid(linewidth=0.4, alpha=0.5)

    axes_flat[-1].axis("off")
    axes[0, 0].legend(fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Demanda (GWh)")
    for ax in axes[1, :3]:
        ax.set_xlabel("Hora")

    fig.suptitle("Partidos de Argentina: perfil observado vs dias comparables", y=1.02)
    fig.tight_layout()
    plt.show()


def plot_election_profiles(
    daily_demand: pd.DataFrame,
    conditions: pd.DataFrame,
) -> None:
    """Plot observed and expected profiles for the 2023 election Sundays."""
    df = _prepare_event_frame(daily_demand, conditions)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

    for ax, event in zip(axes, ELECTION_EVENTS):
        event_row, controls, _ = _control_subset(df, event["fecha"])
        expected_profile = controls[DEMAND_COLUMNS].mean()
        observed_profile = event_row[DEMAND_COLUMNS]

        ax.plot(HOURS, observed_profile / 1000, color="#D65F5F", linewidth=1.8, label="evento")
        ax.plot(HOURS, expected_profile / 1000, color="#4878CF", linewidth=1.8, label="control")
        ax.axvspan(8, 18, color="#F4B400", alpha=0.16, label="votacion")
        ax.set_title(event["evento"])
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.set_xlabel("Hora")
        ax.grid(linewidth=0.4, alpha=0.5)

    axes[0].set_ylabel("Demanda (GWh)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Domingos electorales 2023: perfil observado vs dias comparables", y=1.03)
    fig.tight_layout()
    plt.show()
