import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
PROVINCE_ORDER = [
    'buenos_aires', 'catamarca', 'chaco', 'chubut', 'cordoba',
    'corrientes', 'entre_rios', 'formosa', 'jujuy', 'la_pampa',
    'la_rioja', 'mendoza', 'misiones', 'neuquen', 'rio_negro',
    'salta', 'san_juan', 'san_luis', 'santa_cruz', 'santa_fe',
    'sgo_del_estero', 'tucuman'
]

CLIMATE_FEATURES = [
    'temp_max', 'temp_min', 'temp_media', 'temp_amplitud',
    'sensacion_max', 'sensacion_min', 'sensacion_media', 'sensacion_amplitud',
    'humedad_prom', 'viento_prom',
    'grado_calor', 'grado_frio',
]

SOCIOECONOMIC_FEATURES = [
    'poblacion','pbg_per_capita','participacion_industrial','superficie_km2'
]

CALENDAR_FEATURES = ['mes', 'dia_semana', 'es_feriado']


def load_climate(path: Path) -> pd.DataFrame:
    """Load daily climate data per province and pivot to wide format."""
    df = pd.read_csv(path, parse_dates=['fecha'])
    missing = [f for f in CLIMATE_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Faltan features climáticas: {missing}")
    return df


def load_calendar(path: Path) -> pd.DataFrame:
    """Load daily calendar variables."""
    df = pd.read_csv(path, parse_dates=['fecha'])
    missing = [f for f in CALENDAR_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Faltan features de calendario: {missing}")
    return df[['fecha'] + CALENDAR_FEATURES]


def load_socioeconomic(path: Path) -> pd.DataFrame:
    """
    Load static socioeconomic indicators per province.
    Expected columns: provincia, poblacion, pbi_per_capita,
                      participacion_industrial, superficie_km2
    If file not found, returns a placeholder DataFrame with ones.
    """
    if not path.exists():
        print(f'⚠ No se encontró {path.name} — usando placeholders.')
        print('  Reemplazar con datos reales antes de entrenar el modelo.')
        return pd.DataFrame({
            'provincia': PROVINCE_ORDER,
            **{f: [1.0] * len(PROVINCE_ORDER) for f in SOCIOECONOMIC_FEATURES}
        })
    df = pd.read_csv(path)
    missing = [f for f in SOCIOECONOMIC_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Faltan indicadores socioeconómicos: {missing}")
    return df[['provincia'] + SOCIOECONOMIC_FEATURES]


def pivot_climate(clima: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot climate from long format (fecha, provincia, features)
    to wide format (fecha, temp_max_buenos_aires, temp_max_catamarca, ...).
    Column order follows PROVINCE_ORDER for consistency with x.
    """
    pivot = clima.pivot(index='fecha', columns='provincia', values=CLIMATE_FEATURES)
    # Flatten MultiIndex: (temp_max, buenos_aires) → temp_max_buenos_aires
    pivot.columns = [f'{feat}_{prov}' for feat, prov in pivot.columns]
    # Reorder columns: all features for province 1, then province 2, etc.
    cols_ordered = [
        f'{feat}_{prov}'
        for prov in PROVINCE_ORDER
        for feat in CLIMATE_FEATURES
        if f'{feat}_{prov}' in pivot.columns
    ]
    return pivot[cols_ordered].reset_index()


def normalize_socioeconomic(ind: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize each socioeconomic indicator across provinces."""
    ind_norm = ind.copy()
    for feat in SOCIOECONOMIC_FEATURES:
        min_val = ind[feat].min()
        max_val = ind[feat].max()
        if max_val > min_val:
            ind_norm[feat] = (ind[feat] - min_val) / (max_val - min_val)
    return ind_norm


def build_socioeconomic_row(ind: pd.DataFrame) -> dict:
    """
    Build a dict of socioeconomic features in wide format.
    Keys: poblacion_buenos_aires, pbi_per_capita_buenos_aires, ...
    Values are the same for every day — repeated at join time.
    """
    ind_norm = normalize_socioeconomic(ind).set_index('provincia')
    row = {}
    for prov in PROVINCE_ORDER:
        if prov not in ind_norm.index:
            print(f'  ⚠ Provincia {prov} no encontrada en indicadores')
            continue
        for feat in SOCIOECONOMIC_FEATURES:
            row[f'{feat}_{prov}'] = ind_norm.loc[prov, feat]
    return row


# ─────────────────────────────────────────────
# 1. CARGAR INPUTS
# ─────────────────────────────────────────────
clima    = load_climate(Path('../data/processed/clima_diario_por_provincia.csv'))
calendar = load_calendar(Path('../data/processed/c_calendario.csv'))
ind      = load_socioeconomic(Path('../data/processed/indicadores_provinciales.csv'))

print(f'Clima diario: {clima.shape}')
print(f'Calendario:   {calendar.shape}')
print(f'Indicadores:  {ind.shape}')

# ─────────────────────────────────────────────
# 2. PIVOT CLIMA A FORMATO ANCHO
# ─────────────────────────────────────────────
clima_wide = pivot_climate(clima)
print(f'\nClima pivoteado: {clima_wide.shape}')

# ─────────────────────────────────────────────
# 3. UNIR CALENDARIO
# ─────────────────────────────────────────────
c = calendar.merge(clima_wide, on='fecha', how='inner')
print(f'Después de join con calendario: {c.shape}')

# ─────────────────────────────────────────────
# 4. AGREGAR INDICADORES SOCIOECONÓMICOS
#    Mismo valor para todos los días — se agrega como columnas constantes
# ─────────────────────────────────────────────
ind_row = build_socioeconomic_row(ind)
for col, val in ind_row.items():
    c[col] = val

# ─────────────────────────────────────────────
# 5. ORDENAR COLUMNAS
#    fecha | calendario | clima por provincia | indicadores por provincia
# ─────────────────────────────────────────────
clima_cols = [f'{feat}_{prov}' for prov in PROVINCE_ORDER for feat in CLIMATE_FEATURES if f'{feat}_{prov}' in c.columns]
ind_cols   = [f'{feat}_{prov}' for prov in PROVINCE_ORDER for feat in SOCIOECONOMIC_FEATURES if f'{feat}_{prov}' in c.columns]

c = c[['fecha'] + CALENDAR_FEATURES + clima_cols + ind_cols]
c = c.sort_values('fecha').reset_index(drop=True)

# ─────────────────────────────────────────────
# 6. VALIDAR ALINEACIÓN CON X
# ─────────────────────────────────────────────
x_fechas = pd.read_csv('../data/processed/x_perfiles_diarios_provincias.csv', usecols=['fecha'], parse_dates=['fecha'])
fechas_comunes = set(c['fecha']).intersection(set(x_fechas['fecha']))
print(f'\nFechas en c: {len(c)}')
print(f'Fechas en x: {len(x_fechas)}')
print(f'Fechas comunes: {len(fechas_comunes)}')
if len(fechas_comunes) < len(x_fechas):
    diff = set(x_fechas['fecha']) - set(c['fecha'])
    print(f'⚠ {len(diff)} fechas en x sin datos de clima — revisar rango en armar_clima.py')

# ─────────────────────────────────────────────
# 7. GUARDAR
# ─────────────────────────────────────────────
c.to_csv('../data/processed/c_condiciones_diarias.csv', index=False, float_format='%.6f')

print(f'\n✓ c_condiciones_diarias.csv guardado')
print(f'  Shape: {c.shape}')
print(f'  Composición de c (sin fecha):')
print(f'    Calendario:   {len(CALENDAR_FEATURES)} features')
print(f'    Clima Op. A:  {len(CLIMATE_FEATURES)} × 22 provincias = {len(CLIMATE_FEATURES)*22} features')
print(f'    Indicadores:  {len(SOCIOECONOMIC_FEATURES)} × 22 provincias = {len(SOCIOECONOMIC_FEATURES)*22} features')
print(f'    Total:        {len(CALENDAR_FEATURES) + len(CLIMATE_FEATURES)*22 + len(SOCIOECONOMIC_FEATURES)*22} features')
print(f'\nPrimeras columnas: {c.columns[:8].tolist()}')
print(c.head())