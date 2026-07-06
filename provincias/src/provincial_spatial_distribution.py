import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
import geopandas as gpd

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

GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_ARG_1.json"

GADM_TO_PROVINCE = {
    'BuenosAires':          'buenos_aires',
    'CiudaddeBuenosAires':  'buenos_aires',   # ← se suman al mismo
    'Catamarca':            'catamarca',
    'Chaco':                'chaco',
    'Chubut':               'chubut',
    'Córdoba':              'cordoba',
    'Corrientes':           'corrientes',
    'EntreRíos':            'entre_rios',
    'Formosa':              'formosa',
    'Jujuy':                'jujuy',
    'LaPampa':              'la_pampa',
    'LaRioja':              'la_rioja',
    'Mendoza':              'mendoza',
    'Misiones':             'misiones',
    'Neuquén':              'neuquen',
    'RíoNegro':             'rio_negro',
    'Salta':                'salta',
    'SanJuan':              'san_juan',
    'SanLuis':              'san_luis',
    'SantaCruz':            'santa_cruz',
    'SantaFe':              'santa_fe',
    'SantiagodelEstero':    'sgo_del_estero',
    'TierradelFuego':       None,              # ← no está en el dataset
    'Tucumán':              'tucuman',
}


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

def plot_argentina_map(
    participation: pd.DataFrame
) -> Axes:
    """
    Plot a choropleth map of Argentina colored by each province's
    average share of national electricity demand.

    Requires geopandas. Install with: pip install geopandas fiona pyproj shapely
    Shapefile is downloaded automatically from GADM (no file needed in repo).
    """
    
    PROVINCE_LABELS = {
        'buenos_aires': 'Buenos Aires',
        'catamarca': 'Catamarca',
        'chaco': 'Chaco',
        'chubut': 'Chubut',
        'cordoba': 'Córdoba',
        'corrientes': 'Corrientes',
        'entre_rios': 'Entre Ríos',
        'formosa': 'Formosa',
        'jujuy': 'Jujuy',
        'la_pampa': 'La Pampa',
        'la_rioja': 'La Rioja',
        'mendoza': 'Mendoza',
        'misiones': 'Misiones',
        'neuquen': 'Neuquén',
        'rio_negro': 'Río Negro',
        'salta': 'Salta',
        'san_juan': 'San Juan',
        'san_luis': 'San Luis',
        'santa_cruz': 'Santa Cruz',
        'santa_fe': 'Santa Fe',
        'sgo_del_estero': 'Sgo. del Estero',
        'tucuman': 'Tucumán',
    }

    gdf = gpd.read_file(GADM_URL)
    gdf['provincia'] = gdf['NAME_1'].map(GADM_TO_PROVINCE)
    gdf = gdf.merge(participation, on='provincia', how='left')
 
    # Disolver CABA + Buenos Aires en un solo polígono
    gdf = gdf.dissolve(by='provincia', aggfunc='first').reset_index()
 
    # Layout: mapa a la izquierda, ranking a la derecha
    fig, (ax_map, ax_rank) = plt.subplots(
        1, 2,
        figsize=(13, 11),
        gridspec_kw={'width_ratios': [2, 1]}
    )
 
    gdf.plot(
        column='participacion_pct',
        ax=ax_map,
        cmap='YlOrRd',
        edgecolor='#cccccc',
        linewidth=0.6,
        legend=True,
        legend_kwds={
            'label': 'Participación promedio (%)',
            'orientation': 'horizontal',
            'shrink': 0.7,
            'pad': 0.02,
        },
        missing_kwds={'color': '#e8e8e8', 'label': 'Sin datos'},
    )
 
    ax_map.set_title('Participación promedio en la\ndemanda nacional por provincia', fontsize=12)
    ax_map.axis('off')
 
    # Ranking lateral
    ranked = (
        participation
        .dropna(subset=['participacion_pct'])
        .sort_values('participacion_pct', ascending=True)
        .copy()
    )
    ranked['label'] = ranked['provincia'].map(PROVINCE_LABELS).fillna(ranked['provincia'])
 
    cmap = plt.get_cmap('YlOrRd')
    vmin = ranked['participacion_pct'].min()
    vmax = ranked['participacion_pct'].max()
    colors = [cmap((v - vmin) / (vmax - vmin)) for v in ranked['participacion_pct']]
 
    bars = ax_rank.barh(
        ranked['label'],
        ranked['participacion_pct'],
        color=colors,
        edgecolor='#cccccc',
        linewidth=0.4,
        height=0.7,
    )
 
    # Etiquetas de valor al final de cada barra
    for bar, val in zip(bars, ranked['participacion_pct']):
        ax_rank.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}%',
            va='center',
            fontsize=8,
            color='#333333',
        )
 
    ax_rank.set_xlabel('Participación promedio (%)', fontsize=9)
    ax_rank.set_title('Ranking provincial', fontsize=10)
    ax_rank.spines['top'].set_visible(False)
    ax_rank.spines['right'].set_visible(False)
    ax_rank.tick_params(axis='y', labelsize=8)
    ax_rank.tick_params(axis='x', labelsize=8)
    ax_rank.set_xlim(0, ranked['participacion_pct'].max() * 1.25)
 
    plt.tight_layout()
    plt.savefig('../results/mapa_participacion_provincial.png', dpi=150)
    plt.show()
    return ax_map