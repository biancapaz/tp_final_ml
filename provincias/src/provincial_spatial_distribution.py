import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes


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
SEASON_ORDER = ["Verano", "Otonio", "Invierno", "Primavera"]

TOP_PROVINCES = ["buenos_aires", "cordoba", "santa_fe"]


def compute_province_shares(hourly_demand: pd.DataFrame) -> pd.DataFrame:
    """Compute each province's share of national demand for every hourly record."""
    total = hourly_demand[PROVINCE_ORDER].sum(axis=1)
    return hourly_demand[PROVINCE_ORDER].div(total, axis=0) * 100


def compute_province_participation(hourly_demand: pd.DataFrame) -> pd.DataFrame:
    """Compute each province's average share of national hourly demand."""
    shares = compute_province_shares(hourly_demand)
    participation = shares.mean().rename("participacion_pct").sort_values(ascending=False)
    return participation.reset_index().rename(columns={"index": "provincia"})


def compute_participation_by_season(hourly_demand: pd.DataFrame) -> pd.DataFrame:
    """Compute average province participation for each season of the year."""
    shares = compute_province_shares(hourly_demand)
    shares["estacion"] = hourly_demand["fecha"].dt.month.map(SEASON_MAP)
    return shares.groupby("estacion")[PROVINCE_ORDER].mean().reindex(SEASON_ORDER)


def compute_participation_by_hour(hourly_demand: pd.DataFrame) -> pd.DataFrame:
    """Compute average province participation for each hour of the day."""
    shares = compute_province_shares(hourly_demand)
    shares["hora"] = hourly_demand["hora"]
    return shares.groupby("hora")[PROVINCE_ORDER].mean()


def compute_participation_by_month(hourly_demand: pd.DataFrame) -> pd.DataFrame:
    """Compute average province participation for each calendar month."""
    shares = compute_province_shares(hourly_demand)
    shares["mes"] = hourly_demand["fecha"].dt.month
    return shares.groupby("mes")[PROVINCE_ORDER].mean()


def summarize_top_province_concentration(
    participation: pd.DataFrame,
    top_provinces: list[str] = TOP_PROVINCES,
) -> float:
    """Return the combined average participation share of the given top provinces."""
    mask = participation["provincia"].isin(top_provinces)
    return float(participation.loc[mask, "participacion_pct"].sum())


def plot_participation_barplot(participation: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot the average national demand participation of each province."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))
    ax.bar(participation["provincia"], participation["participacion_pct"], color="#4c78a8")
    ax.set_title("Participacion promedio de cada provincia en la demanda nacional")
    ax.set_xlabel("Provincia")
    ax.set_ylabel("Participacion (%)")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(axis="y", alpha=0.25)
    return ax


def plot_participation_heatmap_by_hour(
    participation_by_hour: pd.DataFrame,
    ax: Axes | None = None,
) -> Axes:
    """Plot a province x hour heatmap of average demand participation."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    data = participation_by_hour[PROVINCE_ORDER].T
    image = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_title("Participacion provincial promedio por hora del dia")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Provincia")
    ax.set_xticks(range(len(participation_by_hour.index)))
    ax.set_xticklabels(participation_by_hour.index)
    ax.set_yticks(range(len(PROVINCE_ORDER)))
    ax.set_yticklabels(PROVINCE_ORDER)
    colorbar = ax.figure.colorbar(image, ax=ax)
    colorbar.set_label("Participacion (%)")
    return ax


def plot_participation_heatmap_by_month(
    participation_by_month: pd.DataFrame,
    ax: Axes | None = None,
) -> Axes:
    """Plot a province x month heatmap of average demand participation."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    data = participation_by_month[PROVINCE_ORDER].T
    image = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_title("Participacion provincial promedio por mes")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Provincia")
    ax.set_xticks(range(len(participation_by_month.index)))
    ax.set_xticklabels([MONTH_NAMES[month - 1] for month in participation_by_month.index])
    ax.set_yticks(range(len(PROVINCE_ORDER)))
    ax.set_yticklabels(PROVINCE_ORDER)
    colorbar = ax.figure.colorbar(image, ax=ax)
    colorbar.set_label("Participacion (%)")
    return ax


def plot_typical_day_stacked_area(
    hourly_demand: pd.DataFrame,
    date: str,
    ax: Axes | None = None,
) -> Axes:
    """Plot hourly demand for one day as a stacked area chart by province."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))
    day_data = hourly_demand.loc[hourly_demand["fecha"] == pd.Timestamp(date)].sort_values("hora")
    ax.stackplot(
        day_data["hora"],
        [day_data[province] for province in PROVINCE_ORDER],
        labels=PROVINCE_ORDER,
    )
    ax.set_title(f"Demanda horaria apilada por provincia — {date}")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Demanda (MWh)")
    ax.set_xticks(HOURS)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    return ax
