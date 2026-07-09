from pathlib import Path

import requests
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

CIUDADES = {
    'buenos_aires': {'lat': -34.6037, 'lon': -58.3816, 'peso': 0.40},
    'cordoba':      {'lat': -31.4135, 'lon': -64.1811, 'peso': 0.15},
    'rosario':      {'lat': -32.9442, 'lon': -60.6505, 'peso': 0.12},
    'mendoza':      {'lat': -32.8908, 'lon': -68.8272, 'peso': 0.10},
    'tucuman':      {'lat': -26.8083, 'lon': -65.2176, 'peso': 0.08},
    'salta':        {'lat': -24.7859, 'lon': -65.4116, 'peso': 0.05},
    'neuquen':      {'lat': -38.9516, 'lon': -68.0591, 'peso': 0.05},
    'resistencia':  {'lat': -27.4606, 'lon': -58.9867, 'peso': 0.05},
}

VARIABLES = 'temperature_2m,relativehumidity_2m,windspeed_10m,apparent_temperature'
START = '2021-01-01'
END   = '2026-04-30'

def descargar_ciudad(nombre, lat, lon):
    url = (
        f'https://archive-api.open-meteo.com/v1/archive'
        f'?latitude={lat}&longitude={lon}'
        f'&start_date={START}&end_date={END}'
        f'&hourly={VARIABLES}'
        f'&timezone=America%2FArgentina%2FBuenos_Aires'
    )
    r = requests.get(url)
    data = r.json()['hourly']
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'])
    df['fecha'] = df['time'].dt.date
    df['hora'] = df['time'].dt.hour + 1  # para que hora vaya de 1 a 24 igual que tu dataset
    df = df.drop(columns=['time'])
    df['ciudad'] = nombre
    print(f'  ✓ {nombre}: {len(df)} filas')
    return df

# Descargar todas las ciudades
dfs = []
for nombre, info in CIUDADES.items():
    print(f'Descargando {nombre}...')
    df = descargar_ciudad(nombre, info['lat'], info['lon'])
    df['peso'] = info['peso']
    dfs.append(df)

df_clima = pd.concat(dfs, ignore_index=True)

# Calcular promedio ponderado por fecha y hora
df_clima['fecha'] = pd.to_datetime(df_clima['fecha'])

clima_ef = df_clima.groupby(['fecha', 'hora']).apply(
    lambda g: pd.Series({
        'temp_ef':     (g['temperature_2m']     * g['peso']).sum() / g['peso'].sum(),
        'humedad_ef':  (g['relativehumidity_2m'] * g['peso']).sum() / g['peso'].sum(),
        'viento_ef':   (g['windspeed_10m']       * g['peso']).sum() / g['peso'].sum(),
        'sensacion_ef':(g['apparent_temperature']* g['peso']).sum() / g['peso'].sum(),
    })
).reset_index()

print(f'\nClima efectivo: {clima_ef.shape}')
print(clima_ef.head())

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
clima_ef.to_csv(INTERIM_DIR / 'clima_efectivo_horario.csv', index=False)
print('✓ data/interim/clima_efectivo_horario.csv guardado')
