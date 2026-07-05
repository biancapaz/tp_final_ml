import unicodedata
import pandas as pd

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
ARCHIVO_1 = "../data/raw/dataset_demanda_horaria_provincia_2019_2022.csv"
ARCHIVO_2 = "../data/raw/dataset_demanda_horaria_provincia_2023_2026.csv"

PROVINCE_ORDER = [
    'buenos_aires', 'catamarca', 'chaco', 'chubut', 'cordoba',
    'corrientes', 'entre_rios', 'formosa', 'jujuy', 'la_pampa',
    'la_rioja', 'mendoza', 'misiones', 'neuquen', 'rio_negro',
    'salta', 'san_juan', 'san_luis', 'santa_cruz', 'santa_fe',
    'sgo_del_estero', 'tucuman'
]
N_HOURS = 24

# Mapeo de nombres originales del CSV a nombres normalizados
COLUMN_MAP = {
    'buenos aires':  'buenos_aires',
    'catamarca':     'catamarca',
    'chaco':         'chaco',
    'chubut':        'chubut',
    'cordoba':       'cordoba',
    'corrientes':    'corrientes',
    'entre rios':    'entre_rios',
    'formosa':       'formosa',
    'jujuy':         'jujuy',
    'la pampa':      'la_pampa',
    'la rioja':      'la_rioja',
    'mendoza':       'mendoza',
    'misiones':      'misiones',
    'neuquen':       'neuquen',
    'rio negro':     'rio_negro',
    'salta':         'salta',
    'san juan':      'san_juan',
    'san luis':      'san_luis',
    'santa cruz':    'santa_cruz',
    'santa fe':      'santa_fe',
    'sgo.del estero': 'sgo_del_estero',
    'tucuman':       'tucuman',
}


def normalize_col(col: str) -> str:
    """Normalize a column name: remove accents, lowercase, underscores."""
    if not isinstance(col, str):
        return str(col)
    col = unicodedata.normalize("NFKD", col)
    col = col.encode("ascii", "ignore").decode("ascii")
    col = col.strip().lower()
    col = col.replace(" ", "_")
    return col


def parse_number(val) -> float:
    """Parse Argentine number format: dots as thousands sep, comma as decimal."""
    if pd.isna(val):
        return float('nan')
    s = str(val).strip().replace('.', '').replace(',', '.')
    try:
        return float(s)
    except ValueError:
        return float('nan')


def load_csv(path: str) -> pd.DataFrame:
    """Load a CAMMESA provincial demand CSV with semicolon separator and 3 header rows."""
    df = pd.read_csv(
        path,
        sep=';',
        skiprows=3,        # skip the 3 title rows
        encoding='latin-1',
        header=0,
        dtype=str,         # read everything as string first — parse numbers manually
    )
    # Drop fully empty columns
    df = df.dropna(axis=1, how='all')
    df.columns = [c.strip() for c in df.columns]
    return df


# ─────────────────────────────────────────────
# 1. CARGAR
# ─────────────────────────────────────────────
df1 = load_csv(ARCHIVO_1)
df2 = load_csv(ARCHIVO_2)

print("Columnas dataset 1:", df1.columns.tolist())
print("Columnas dataset 2:", df2.columns.tolist())


def process_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Clean and normalize a raw CAMMESA provincial demand dataframe.
    Returns a long-format dataframe with columns:
    fecha, hora, dia_semana, es_feriado, + one column per province.
    """
    # Identify key columns — normalize keys to handle encoding issues (AÃO, NÂ° MES)
    import unicodedata as _ud
    def _nk(s):
        s = _ud.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        return s.strip().lower()
    col_lower = {_nk(c): c for c in df.columns}

    hora_col  = col_lower.get('hora')
    tipod_col = col_lower.get('tipo dia')

    # Use AÑO + N° MES + N° DIA to build date — avoids any format ambiguity
    ano_col  = col_lower.get('aao')       # AÃO normalizes to aao
    nmes_col = col_lower.get('na mes')    # NÂ° MES normalizes to na mes
    ndia_col = col_lower.get('na dia')    # NÂ° DIA normalizes to na dia

    if not all([hora_col, tipod_col, ano_col, nmes_col, ndia_col]):
        raise KeyError(f"[{dataset_name}] No se encontraron columnas clave. Disponibles: {df.columns.tolist()}")

    # Build fecha from numeric columns — no string format ambiguity
    df['fecha_parsed'] = pd.to_datetime({
        'year':  pd.to_numeric(df[ano_col].str.strip(),  errors='coerce'),
        'month': pd.to_numeric(df[nmes_col].str.strip(), errors='coerce'),
        'day':   pd.to_numeric(df[ndia_col].str.strip(), errors='coerce'),
    }, errors='coerce')

    # Parse hora
    df['hora_parsed'] = pd.to_numeric(df[hora_col].str.strip(), errors='coerce').astype('Int64')

    # Build es_feriado: No Hábil on a weekday (not Sat/Sun)
    df['dia_semana'] = df['fecha_parsed'].dt.dayofweek  # 0=Mon, 6=Sun
    tipo_lower = df[tipod_col].str.lower().str.strip()
    df['es_feriado'] = (
        tipo_lower.str.contains('no') &
        ~df['dia_semana'].isin([5, 6])
    ).astype(int)

    # Identify province columns using COLUMN_MAP
    province_cols = {}
    for raw_col in df.columns:
        key = raw_col.strip().lower()
        if key in COLUMN_MAP:
            province_cols[raw_col] = COLUMN_MAP[key]

    missing = [p for p in PROVINCE_ORDER if p not in province_cols.values()]
    if missing:
        print(f"  ⚠ [{dataset_name}] Provincias no encontradas: {missing}")

    # Parse province demand values
    result = df[['fecha_parsed', 'hora_parsed', 'dia_semana', 'es_feriado']].copy()
    result.columns = ['fecha', 'hora', 'dia_semana', 'es_feriado']

    for raw_col, prov_name in province_cols.items():
        result[prov_name] = df[raw_col].apply(parse_number)

    # Drop rows with invalid fecha or hora
    result = result.dropna(subset=['fecha', 'hora'])
    result['hora'] = result['hora'].astype(int)

    return result


# ─────────────────────────────────────────────
# 2. PROCESAR CADA DATASET
# ─────────────────────────────────────────────
df1_clean = process_dataset(df1, "2019-2022")
df2_clean = process_dataset(df2, "2023-2026")

print(f"\nDataset 1 procesado: {df1_clean.shape}, rango: {df1_clean['fecha'].min()} → {df1_clean['fecha'].max()}")
print(f"Dataset 2 procesado: {df2_clean.shape}, rango: {df2_clean['fecha'].min()} → {df2_clean['fecha'].max()}")

# ─────────────────────────────────────────────
# 3. RESOLVER OVERLAP Y EXCLUIR 2020
# ─────────────────────────────────────────────
fecha_inicio_df2 = df2_clean['fecha'].min()
df1_clean = df1_clean[df1_clean['fecha'] < fecha_inicio_df2]

df_base = pd.concat([df1_clean, df2_clean], ignore_index=True)
df_base = df_base.sort_values(['fecha', 'hora']).reset_index(drop=True)

# Excluir 2020 — comportamiento anómalo por pandemia
df_base = df_base[df_base['fecha'].dt.year != 2020]

print(f"\nDataset base: {df_base.shape[0]} filas")
print(f"Rango final: {df_base['fecha'].min()} → {df_base['fecha'].max()}")
print(f"Años presentes: {sorted(df_base['fecha'].dt.year.unique().tolist())}")

# ─────────────────────────────────────────────
# 4. CONSTRUIR X — perfil diario por provincia
#    Columnas: {provincia}_{hora:02d}
#    Orden de provincias según PROVINCE_ORDER
# ─────────────────────────────────────────────
x_parts = []
for provincia in PROVINCE_ORDER:
    if provincia not in df_base.columns:
        print(f"  ⚠ Provincia {provincia} no encontrada en df_base, saltando.")
        continue
    pivot = df_base.pivot(index='fecha', columns='hora', values=provincia)
    pivot.columns = [f'{provincia}_{int(h):02d}' for h in pivot.columns]
    x_parts.append(pivot)

x = pd.concat(x_parts, axis=1).reset_index()

n_prov = len(x_parts)
print(f"\nX (perfiles diarios): {x.shape}")
print(f"  {n_prov} provincias × {N_HOURS} horas = {n_prov * N_HOURS} columnas de demanda")
print(x.head())

# ─────────────────────────────────────────────
# 5. CONSTRUIR CALENDARIO para C
# ─────────────────────────────────────────────
df_base['mes'] = df_base['fecha'].dt.month

calendario = df_base.groupby('fecha').agg(
    mes=('mes', 'first'),
    dia_semana=('dia_semana', 'first'),
    es_feriado=('es_feriado', 'first')
).reset_index()

print(f"\nCalendario: {calendario.shape}")
print(calendario.head())
print("\nDistribución es_feriado:", calendario['es_feriado'].value_counts().to_dict())
print("Distribución dia_semana:", calendario['dia_semana'].value_counts().sort_index().to_dict())

# ─────────────────────────────────────────────
# 6. GUARDAR
# ─────────────────────────────────────────────
x.to_csv("../data/processed/x_perfiles_diarios_provincias.csv", index=False)
calendario.to_csv("../data/processed/c_calendario.csv", index=False)
df_base.to_csv("../data/processed/dataset_base_horario.csv", index=False)

print("\n✓ Archivos guardados en data/processed/:")
print("  - dataset_base_horario.csv  → base horaria completa por provincia")
print(f"  - x_perfiles_diarios_provincias.csv    → {x.shape[0]} días × {x.shape[1]-1} columnas de demanda")
print("  - c_calendario.csv          → variables de calendario")