# Trabajo Practico Final - Machine Learning

## Ejecucion del modelo final

El modelo final seleccionado para la entrega es **CVAE-A**. Es un modelo
generativo condicional que produce escenarios horarios de demanda electrica
provincial para el sistema completo:

```text
22 provincias x 24 horas = 528 valores por dia
```

Por tratarse de un modelo generativo, el resultado principal no es una unica
prediccion deterministica sino un conjunto de escenarios. El script guarda la
mediana de los escenarios como prediccion central y los percentiles P5 y P95
como banda de incertidumbre.

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar el modelo final:

```bash
python run_final_model.py \
    --input provincias/data/processed/c_condiciones_diarias.csv \
    --output predictions/ \
    --seed 42 \
    --n-samples 500 \
    --device cpu
```

Argumentos principales:

- `--input`: CSV de entrada con el mismo formato ancho usado para CVAE-A:
  una fila por dia, columna `fecha` y las columnas de contexto de
  `provincias/data/processed/c_condiciones_diarias.csv`. Si el archivo tambien
  incluye las 528 columnas de demanda de
  `provincias/data/processed/x_perfiles_diarios_provincias.csv`, el script
  calcula metricas automaticamente.
- `--output`: carpeta donde se guardan predicciones, muestras, metricas y
  metadata de la corrida.
- `--seed`: semilla aleatoria. El default es `42`.
- `--n-samples`: cantidad de escenarios por dia. El default es `500`.
- `--device`: dispositivo PyTorch. El default es `cpu` para favorecer
  reproducibilidad.
- `--artifacts-dir`: carpeta de artifacts del modelo final. El default es
  `models/final_cvae_a/`.

Artifacts esperados en `models/final_cvae_a/`:

- `cvae_a_final.pt`
- `x_scaler_a.pkl`
- `c_scaler_a.pkl`
- `cvae_a_config.json`
- `cvae_a_metadata.json`

Outputs generados:

- `cvae_a_median.csv`: prediccion central en MWh, formato largo con columnas
  `fecha`, `provincia`, `hora`, `median_mwh`.
- `cvae_a_p05.csv`: percentil 5 en MWh, con columna `p05_mwh`.
- `cvae_a_p95.csv`: percentil 95 en MWh, con columna `p95_mwh`.
- `cvae_a_samples.npy`: todos los escenarios generados con shape
  `(n_days, n_samples, 22, 24)`.
- `metrics_test.json`: se crea solo si el input contiene demanda real.
- `run_metadata.json`: metadata de la corrida, seed, paths y dimensiones.

Las metricas de test incluyen `coverage_p5_p95`, `sharpness_mwh`,
`rmse_mean_profile_mwh`, `mae_median_mwh` y `mmd` si la funcion reutilizable de
MMD esta disponible en el repositorio.

Al tratarse de un modelo generativo, las muestras dependen de la semilla
aleatoria. Para reproducir los resultados reportados en el informe se fija
seed=42 por default.

Pueden existir pequeñas diferencias numericas entre entornos por versiones de
PyTorch, BLAS o hardware, especialmente si se usa GPU. Para maximizar la
reproducibilidad de la entrega, el comando recomendado usa `--device cpu`.

## Estructura del repositorio

```text
models/                     artifacts del modelo final y checkpoints
notebooks/                  notebooks de exploracion, entrenamiento y evaluacion
notes/                      consignas, notas y documentacion auxiliar
src/                        codigo reutilizable para analisis y modelos nacionales
results/                    figuras y resultados del informe nacional
provincias/                 pipeline, datos y resultados provinciales
data/raw/                   datos originales o casi originales
data/interim/               datos intermedios de preparacion
data/processed/             datasets procesados para modelado o EDA
scripts/data_preparation/   scripts para reconstruir datasets nacionales
run_final_model.py          script ejecutable del modelo final CVAE-A
```

Datos nacionales reorganizados:

- `data/raw/`: planillas fuente `dataset_2021_2023.xlsx` y
  `dataset_2023_2026.xlsx`.
- `data/interim/`: calendario, condiciones diarias, clima efectivo horario y
  archivos auxiliares intermedios.
- `data/processed/`: `dataset_base_horario.csv`,
  `x_perfiles_diarios.csv` e `indicadores_provinciales.csv`.

Los scripts de `scripts/data_preparation/` usan rutas absolutas calculadas a
partir de la raiz del proyecto, por lo que pueden ejecutarse desde el root del
repo sin copiar datos a la carpeta principal.
