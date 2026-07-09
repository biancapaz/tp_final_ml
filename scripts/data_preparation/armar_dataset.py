from pathlib import Path
import unicodedata

import pandas as pd

# ─────────────────────────────────────────────
# CONFIGURACIÓN — ajustá los nombres de archivo
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ARCHIVO_1 = RAW_DIR / "dataset_2021_2023.xlsx"   # columnas por región 2021-2023
ARCHIVO_2 = RAW_DIR / "dataset_2023_2026.xlsx"   # columnas por provincia 2023-2026

# ─────────────────────────────────────────────
# Helper de normalización de columnas
# ─────────────────────────────────────────────

def normalize_col(col):
    if not isinstance(col, str):
        return str(col)
    normalized = unicodedata.normalize("NFKD", col)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.strip().lower()
    normalized = normalized.replace(" ", "_")
    normalized = normalized.replace(".", "")
    normalized = normalized.replace("-", "_")
    normalized = normalized.replace("%", "pct")
    normalized = normalized.replace("°", "")
    return normalized

# ─────────────────────────────────────────────
# 1. CARGAR LOS DOS ARCHIVOS
# ─────────────────────────────────────────────
# Los archivos pueden tener filas de encabezado extra. Detectamos la fila
# que contiene las columnas clave en lugar de asumir una posición fija.

def find_header_row(path, required_columns, max_rows=20):
    raw = pd.read_excel(path, header=None, nrows=max_rows)
    required_set = {normalize_col(c) for c in required_columns}
    for i, row in raw.iterrows():
        normalized = {normalize_col(v) for v in row.tolist() if pd.notna(v)}
        if required_set.issubset(normalized):
            return i
    raise ValueError(f"No se encontró fila de encabezado en {path} con columnas {required_columns}")

header1 = find_header_row(ARCHIVO_1, ['AÑO', 'MES', 'DÍA', 'TIPO DÍA', 'HORA'])
header2 = find_header_row(ARCHIVO_2, ['AÑO', 'MES', 'N° MES', 'N° DIA', 'TIPO DÍA', 'DÍA', 'FECHA', 'HORA'])

df1 = pd.read_excel(ARCHIVO_1, header=header1)
df2 = pd.read_excel(ARCHIVO_2, header=header2)

df1.columns = [normalize_col(c) for c in df1.columns]
df2.columns = [normalize_col(c) for c in df2.columns]

print("Columnas dataset 1:", df1.columns.tolist())
print("Columnas dataset 2:", df2.columns.tolist())

# ─────────────────────────────────────────────
# 2. NORMALIZAR DATASET 1 (2021-2023)
#    Tiene columnas de demanda regionales → sumamos para TOTAL
# ─────────────────────────────────────────────

# Columnas que no son demanda
meta_cols_1 = {'ano', 'mes', 'dia', 'tipo_dia', 'hora'}
exclude_cols_1 = {'total', 'crono', 'orden', 'supera_200_hs', 'crono1', 'orden1', 'supera_200_hs1'}

# Columnas de demanda válidas para sumar
cols_demanda_1 = [
    c for c in df1.columns
    if c not in meta_cols_1 and c not in exclude_cols_1 and not c.startswith('unnamed')
]

if not cols_demanda_1:
    raise ValueError("No se pudieron identificar columnas de demanda en dataset 1")

# Convertir a numérico y sumar filas
for col in cols_demanda_1:
    df1[col] = pd.to_numeric(df1[col], errors='coerce')

df1['total'] = df1[cols_demanda_1].sum(axis=1)

# Seleccionar columnas de salida
if 'dia' not in df1.columns and 'mes' not in df1.columns:
    raise KeyError('No se encontró la columna DIA o MES en dataset 1 después de normalizar columnas')

if 'tipo_dia' not in df1.columns:
    raise KeyError('No se encontró la columna TIPO DÍA en dataset 1 después de normalizar columnas')

if 'hora' not in df1.columns:
    raise KeyError('No se encontró la columna HORA en dataset 1 después de normalizar columnas')

fecha_col_1 = 'dia' if 'dia' in df1.columns else 'mes'

df1_clean = df1[[fecha_col_1, 'hora', 'tipo_dia', 'total']].copy()
df1_clean.columns = ['fecha', 'hora', 'tipo_dia_raw', 'demanda_mwh']
df1_clean['fecha'] = pd.to_datetime(df1_clean['fecha'], errors='coerce')

# Dataset 1: tipo_dia viene como "Hábil", "Sábado", "Domingo o Feriado"
# es_feriado = 1 solo si dice feriado Y no es fin de semana natural
df1_clean['dia_semana'] = df1_clean['fecha'].dt.dayofweek  # 0=lunes, 6=domingo
df1_clean['es_feriado'] = (
    df1_clean['tipo_dia_raw'].str.lower().str.contains('feriado') &
    ~df1_clean['dia_semana'].isin([5, 6])  # no sábado ni domingo
).astype(int)

# ─────────────────────────────────────────────
# 3. NORMALIZAR DATASET 2 (2023-2026)
#    Ya tiene columna TOTAL directamente
#    y columnas separadas: tipo_dia (habil/no habil) y dia (lunes, martes...)
# ─────────────────────────────────────────────

# Verificá con print(df2.columns.tolist()) si estos nombres son correctos
COL_HABIL = 'tipo_dia'   # columna con "Hábil" / "No Hábil"
COL_DIA   = 'dia'        # columna con "Lunes", "Martes", "Domingo"...

for col in ['fecha', 'hora', COL_HABIL, COL_DIA, 'total']:
    if col not in df2.columns:
        raise KeyError(f'No se encontró la columna {col.upper()} en dataset 2. Columnas disponibles: {df2.columns.tolist()}')

df2_clean = df2[['fecha', 'hora', COL_HABIL, COL_DIA, 'total']].copy()
df2_clean.columns = ['fecha', 'hora', 'tipo_dia_raw', 'dia_nombre', 'demanda_mwh']
df2_clean['fecha'] = pd.to_datetime(df2_clean['fecha'], errors='coerce')

# Dataset 2: es_feriado = No Hábil en un día que no es sábado ni domingo
df2_clean['dia_semana'] = df2_clean['fecha'].dt.dayofweek
df2_clean['es_feriado'] = (
    df2_clean['tipo_dia_raw'].str.lower().str.contains('no') &
    ~df2_clean['dia_semana'].isin([5, 6])
).astype(int)

# ─────────────────────────────────────────────
# 4. RESOLVER EL OVERLAP DE 2023
#    El año 2023 aparece en los dos datasets.
#    Nos quedamos con la versión del dataset 2 para ese año
#    porque tiene la estructura más consistente.
# ─────────────────────────────────────────────

# Quedarse solo con las columnas limpias antes de concatenar
cols_finales = ['fecha', 'hora', 'demanda_mwh', 'dia_semana', 'es_feriado']
df1_clean = df1_clean[cols_finales]
df2_clean = df2_clean[cols_finales]

# Resolver overlap: usar df2 desde su primera fecha
fecha_inicio_df2 = df2_clean['fecha'].min()
df1_clean = df1_clean[df1_clean['fecha'] < fecha_inicio_df2]

# ─────────────────────────────────────────────
# 5. CONCATENAR
# ─────────────────────────────────────────────

df_base = pd.concat([df1_clean, df2_clean], ignore_index=True)
df_base = df_base.sort_values(['fecha', 'hora']).reset_index(drop=True)

print(f"\nDataset base: {df_base.shape[0]} filas")
print(f"Rango: {df_base['fecha'].min()} → {df_base['fecha'].max()}")
print(df_base.head())

# ─────────────────────────────────────────────
# 6. CONSTRUIR X — pivot a perfil diario de 24 horas
# ─────────────────────────────────────────────

x = df_base.pivot(index='fecha', columns='hora', values='demanda_mwh')
x.columns = [f'demanda_{int(h):02d}' for h in x.columns]
x = x.reset_index()

print(f"\nX (perfiles diarios): {x.shape}")
print(x.head())

# ─────────────────────────────────────────────
# 7. CONSTRUIR CALENDARIO para C
#    dia_semana y es_feriado ya están en df_base
#    solo agregamos mes y tomamos un valor por día
# ─────────────────────────────────────────────

df_base['mes'] = df_base['fecha'].dt.month

calendario = df_base.groupby('fecha').agg(
    mes=('mes', 'first'),
    dia_semana=('dia_semana', 'first'),
    es_feriado=('es_feriado', 'first')
).reset_index()

print(f"\nCalendario: {calendario.shape}")
print(calendario.head())
print("\nDistribución es_feriado:")
print(calendario['es_feriado'].value_counts())
print("\nDistribución dia_semana:")
print(calendario['dia_semana'].value_counts().sort_index())

# ─────────────────────────────────────────────
# 8. GUARDAR
# ─────────────────────────────────────────────

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

x.to_csv(PROCESSED_DIR / "x_perfiles_diarios.csv", index=False)
calendario.to_csv(INTERIM_DIR / "c_calendario.csv", index=False)
df_base.to_csv(PROCESSED_DIR / "dataset_base_horario.csv", index=False)

print("\n✓ Archivos guardados:")
print("  - data/processed/dataset_base_horario.csv  → base horaria completa")
print("  - data/processed/x_perfiles_diarios.csv    → perfiles de 24h (input del modelo)")
print("  - data/interim/c_calendario.csv            → variables de calendario (parte de c)")
print("\nPróximo paso: agregar clima efectivo a data/interim/c_calendario.csv")
