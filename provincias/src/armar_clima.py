import requests
import pandas as pd
import time

# ─────────────────────────────────────────────
# Una estación representativa por provincia
# Nombres deben coincidir con PROVINCE_ORDER en armar_dataset.py
# ─────────────────────────────────────────────
PROVINCIAS = {
    'buenos_aires':   {'lat': -34.6037, 'lon': -58.3816},
    'catamarca':      {'lat': -28.4696, 'lon': -65.7795},
    'chaco':          {'lat': -27.4606, 'lon': -58.9867},
    'chubut':         {'lat': -43.3002, 'lon': -65.1023},
    'cordoba':        {'lat': -31.4135, 'lon': -64.1811},
    'corrientes':     {'lat': -27.4692, 'lon': -58.8306},
    'entre_rios':     {'lat': -31.7413, 'lon': -60.5238},
    'formosa':        {'lat': -26.1775, 'lon': -58.1781},
    'jujuy':          {'lat': -24.1858, 'lon': -65.2995},
    'la_pampa':       {'lat': -36.6167, 'lon': -64.2833},
    'la_rioja':       {'lat': -29.4131, 'lon': -66.8558},
    'mendoza':        {'lat': -32.8908, 'lon': -68.8272},
    'misiones':       {'lat': -27.3671, 'lon': -55.8965},
    'neuquen':        {'lat': -38.9516, 'lon': -68.0591},
    'rio_negro':      {'lat': -40.8135, 'lon': -63.0000},
    'salta':          {'lat': -24.7859, 'lon': -65.4116},
    'san_juan':       {'lat': -31.5375, 'lon': -68.5364},
    'san_luis':       {'lat': -33.3022, 'lon': -66.3371},
    'santa_cruz':     {'lat': -51.6230, 'lon': -69.2168},
    'santa_fe':       {'lat': -31.6333, 'lon': -60.7000},
    'sgo_del_estero': {'lat': -27.7951, 'lon': -64.2615},
    'tucuman':        {'lat': -26.8083, 'lon': -65.2176},
}

VARIABLES = 'temperature_2m,relativehumidity_2m,windspeed_10m,apparent_temperature'
START = '2019-01-01'
END   = '2026-04-30'

# Umbrales para grado_calor y grado_frio
# Representan el punto a partir del cual se enciende AC o calefacción
HEAT_THRESHOLD = 22.0
COLD_THRESHOLD = 15.0


def descargar_provincia(nombre: str, lat: float, lon: float) -> pd.DataFrame:
    """Download hourly climate data for a single province from Open-Meteo archive API."""
    url = (
        f'https://archive-api.open-meteo.com/v1/archive'
        f'?latitude={lat}&longitude={lon}'
        f'&start_date={START}&end_date={END}'
        f'&hourly={VARIABLES}'
        f'&timezone=America%2FArgentina%2FBuenos_Aires'
    )
    for intento in range(5):
        r = requests.get(url)
        if r.status_code == 429:
            espera = 30 * (intento + 1)
            print(f'  ⚠ Rate limit, esperando {espera}s...')
            time.sleep(espera)
            continue
        r.raise_for_status()
        break
    data = r.json()['hourly']
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'])
    df['fecha'] = df['time'].dt.date
    df['hora'] = df['time'].dt.hour + 1  # 1 a 24, igual que dataset de demanda
    df = df.drop(columns=['time'])
    df['provincia'] = nombre
    df = df.rename(columns={
        'temperature_2m':      'temp',
        'relativehumidity_2m': 'humedad',
        'windspeed_10m':       'viento',
        'apparent_temperature':'sensacion',
    })
    print(f'  ✓ {nombre}: {len(df)} filas')
    return df


# ─────────────────────────────────────────────
# Descargar todas las provincias
# ─────────────────────────────────────────────
dfs = []
for nombre, info in PROVINCIAS.items():
    print(f'Descargando {nombre}...')
    df = descargar_provincia(nombre, info['lat'], info['lon'])
    dfs.append(df)
    time.sleep(2)  # pausa para no saturar la API

df_clima = pd.concat(dfs, ignore_index=True)
df_clima['fecha'] = pd.to_datetime(df_clima['fecha'])

# Excluir 2020 — consistente con armar_dataset.py
df_clima = df_clima[df_clima['fecha'].dt.year != 2020]

print(f'\nClima combinado: {df_clima.shape}')
print(f'Años presentes: {sorted(df_clima["fecha"].dt.year.unique().tolist())}')
print(df_clima.head())

# ─────────────────────────────────────────────
# Guardar formato largo — para Opción B (horario)
# Una fila por fecha-hora-provincia
# ─────────────────────────────────────────────
df_clima.to_csv('../data/processed/clima_por_provincia_horario.csv', index=False)
print('✓ clima_por_provincia_horario.csv guardado')

# ─────────────────────────────────────────────
# Opción A — agregado diario por provincia
# Una fila por fecha-provincia
# ─────────────────────────────────────────────
clima_diario = df_clima.groupby(['fecha', 'provincia']).agg(
    temp_max=       ('temp',      'max'),
    temp_min=       ('temp',      'min'),
    temp_media=     ('temp',      'mean'),
    sensacion_max=  ('sensacion', 'max'),
    sensacion_min=  ('sensacion', 'min'),
    sensacion_media=('sensacion', 'mean'),
    humedad_prom=   ('humedad',   'mean'),
    viento_prom=    ('viento',    'mean'),
).reset_index()

# Features derivadas
clima_diario['temp_amplitud']      = clima_diario['temp_max'] - clima_diario['temp_min']
clima_diario['sensacion_amplitud'] = clima_diario['sensacion_max'] - clima_diario['sensacion_min']
clima_diario['grado_calor']        = (clima_diario['temp_media'] - HEAT_THRESHOLD).clip(lower=0)
clima_diario['grado_frio']         = (COLD_THRESHOLD - clima_diario['temp_media']).clip(lower=0)

print(f'\nClima diario por provincia: {clima_diario.shape}')
print(f'Features: {[c for c in clima_diario.columns if c not in ["fecha", "provincia"]]}')
print(clima_diario.head())

clima_diario.to_csv('../data/processed/clima_diario_por_provincia.csv', index=False)
print('✓ clima_diario_por_provincia.csv guardado')

print('\nArchivos guardados en data/processed/:')
print('  - clima_por_provincia_horario.csv  → Opción B (clima horario por provincia)')
print('  - clima_diario_por_provincia.csv   → Opción A (clima diario, 10 features por provincia)')