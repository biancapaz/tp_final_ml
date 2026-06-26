# Planificacion de modelos y experimentos

## Objetivo

El objetivo de esta etapa es construir una comparacion metodologicamente completa
de modelos para generar perfiles horarios diarios de demanda electrica
condicionados por variables de calendario y clima.

El baseline actual queda definido como:

- Modelo: CVAE MLP.
- Entrada condicional: `C-full`, 21 variables.
- Salida: perfil diario de 24 horas en MWh.
- Split temporal: train 2021-2023, validation 2024, test 2025-2026.
- Escalado: ajustado solo con train para evitar leakage.
- Entrenamiento: multi-seed, early stopping, beta warm-up.
- Evaluacion: reconstruccion y generacion separadas.

La idea no es probar modelos al azar, sino organizar una escalera experimental
donde cada variante responda una pregunta concreta.

## Protocolo comun para todos los experimentos

Para que la comparacion sea justa, todos los modelos deberian respetar el mismo
protocolo experimental:

- Usar el mismo split temporal:
  - `train`: 2021-2023.
  - `validation`: 2024.
  - `test`: 2025-2026.
- Ajustar cualquier scaler, PCA, codificador o imputador solo con train.
- Elegir hiperparametros y checkpoints usando validation.
- Reportar test solo al final.
- Repetir los modelos principales con al menos 3 seeds.
- Guardar configuracion, metricas y outputs por experimento.
- Evaluar tanto error punto a punto como realismo estadistico.

## Metricas de evaluacion

Las metricas deben medir mas que MSE, porque el objetivo es generativo.

### Metricas de error horario

- MAE horario en MWh.
- RMSE horario en MWh.
- MSE estandarizado, para comparar con la loss.

### Metricas de energia y nivel diario

- Error absoluto de energia diaria.
- MAPE de energia diaria.
- Error de demanda media diaria.
- Error de demanda maxima diaria.

### Metricas de forma

- Error de hora pico.
- Accuracy de hora pico.
- Error de rampas hora a hora.
- Error de perfil normalizado por energia o media diaria.

### Metricas por condicion

- Perfil medio real vs generado por tipo de dia.
- Perfil medio real vs generado por mes.
- Percentiles P25, P50, P75 por tipo de dia.
- Error separado para habil, sabado, domingo y feriado.

### Metricas de diversidad

- Desvio horario generado vs real.
- Ratio de variabilidad generada / real.
- Cobertura de bandas percentiles.
- Deteccion de subdispersion, es decir, perfiles generados demasiado suaves.

## Bloque 1: baselines simples obligatorios

Estos modelos sirven como piso de comparacion. Son importantes porque muestran
si los modelos avanzados realmente agregan valor.

| Modelo | Descripcion | Pregunta que responde |
|---|---|---|
| Perfil promedio global | Genera siempre el promedio historico de 24 horas | Cuanto mejora cualquier condicionamiento frente a una regla trivial |
| Promedio por tipo de dia | Promedio historico para habil, sabado, domingo, feriado | Cuanto aporta el calendario basico |
| Promedio por tipo de dia + mes | Promedio por categoria calendario-estacional | Que tan fuerte es un baseline interpretable |
| KNN de dias similares | Busca dias parecidos por tipo de dia, mes y clima | Si copiar dias historicos similares alcanza |
| MLP deterministico | Predice 24 horas desde `C-full` | Cuanto aporta un modelo deep no generativo |
| Random Forest multi-output | Predice 24 horas desde `C-full` | Baseline clasico robusto e interpretable |
| XGBoost / Gradient Boosting multi-output | Predice 24 horas desde `C-full` | Baseline clasico fuerte para variables tabulares |

Resultado esperado:

- Si un CVAE no mejora a estos modelos en metricas generativas, no conviene
  venderlo como modelo superior.
- Si el CVAE iguala o supera el promedio condicionado y el MLP, se justifica
  el enfoque generativo.

## Bloque 2: ablations del condicionamiento

Estas variantes prueban que variables de `C` realmente importan.

| Variante | Condiciones usadas | Pregunta |
|---|---|---|
| CVAE C-full | Todas las 21 variables actuales | Baseline final completo |
| CVAE C-minimal | `tipo_dia`, `mes`, `grado_calor`, `grado_frio` | C-full aporta informacion o mete ruido |
| CVAE calendar-only | Tipo de dia, mes, dia semana, feriado | Cuanto explica el calendario sin clima |
| CVAE weather-only | Variables climaticas diarias | Cuanto explica el clima sin calendario |
| CVAE sin tipo_dia | C-full sin one-hot de tipo de dia | Cuanto depende la forma horaria del tipo de dia |
| CVAE sin clima | C-full sin variables meteorologicas | Cuanto se pierde sin temperatura/sensacion |
| CVAE solo temperatura no lineal | `grado_calor`, `grado_frio` y calendario | Si las variables no lineales alcanzan |
| CVAE con eventos | C-full + flags de eventos | Si eventos especiales ayudan a dias raros |
| CVAE con clima horario | Calendario + clima por hora | Si el efecto climatico horario mejora picos y rampas |

Eventos candidatos para agregar como flags:

- Navidad.
- Anio Nuevo.
- Feriados especiales.
- Final del Mundial 2022.
- Partidos importantes del Mundial 2022.
- Elecciones 2023.
- Periodos COVID/restricciones.
- Apagon 2023-03-01, si se decide incluirlo con etiqueta.

Resultado esperado:

- Si C-minimal se acerca mucho a C-full, se puede argumentar que el modelo
  necesita pocas variables bien elegidas.
- Si C-full mejora picos, energia o diversidad, se justifica mantener todas las
  variables.
- Si weather-only falla en forma y calendar-only falla en extremos climaticos,
  se demuestra complementariedad entre calendario y clima.

## Bloque 3: representacion de la demanda

Estas variantes son muy importantes porque el EDA mostro que nivel y forma no
son exactamente el mismo problema.

| Variante | Idea | Ventaja |
|---|---|---|
| CVAE raw profile | Genera directamente las 24 demandas en MWh | Baseline actual |
| CVAE normalized shape | Genera perfil normalizado por media o energia | Se concentra en la forma |
| Modelo de nivel + CVAE shape | Un modelo predice energia diaria y el CVAE genera forma | Separa nivel y forma |
| CVAE level-shape conjunto | Decoder produce nivel y forma en una arquitectura unica | Variante avanzada y alineada con el EDA |
| CVAE residual | Primero se estima perfil esperado y el CVAE genera residuo | Mejora variabilidad y reduce carga del decoder |
| CVAE de rampas | Genera diferencias hora a hora y reconstruye niveles | Controla dinamica horaria |
| CVAE sobre PCA | PCA de perfiles, CVAE genera componentes, luego inversa PCA | Reduccion de dimension interpretable |

Experimento recomendado:

1. Entrenar el baseline raw profile.
2. Entrenar CVAE normalized shape.
3. Entrenar un modelo de nivel, por ejemplo XGBoost o MLP, para energia diaria.
4. Combinar nivel + shape.
5. Comparar energia, picos, rampas y perfiles medios.

Hipotesis:

- El modelo level-shape deberia mejorar la generacion porque no obliga al mismo
  decoder a aprender simultaneamente escala absoluta y forma relativa.

## Bloque 4: arquitectura del CVAE

Estas variantes prueban si el problema necesita mas capacidad o una arquitectura
mas adecuada para series horarias.

| Variante | Cambios | Pregunta |
|---|---|---|
| MLP chico | hidden 64, latent 4 u 8 | El baseline esta sobredimensionado |
| MLP mediano | hidden 128, latent 8 | Referencia actual |
| MLP grande | hidden 256 o 512, latent 16 | Mas capacidad mejora generacion |
| CVAE con dropout | Dropout en encoder/decoder | Reduce overfitting |
| CVAE con LayerNorm | Normalizacion en capas ocultas | Estabiliza entrenamiento |
| CVAE con BatchNorm | BatchNorm en MLP | Mejora optimizacion |
| Conv1D-CVAE | Encoder/decoder convolucional sobre 24 horas | Captura picos, valles y rampas |
| GRU-CVAE | Decoder secuencial hora por hora | Modela dependencia temporal |
| LSTM-CVAE | Variante recurrente mas expresiva | Captura memoria horaria |
| Transformer-CVAE | Atencion sobre las 24 horas | Modelo avanzado para dependencias globales |

Recomendacion practica:

- Probar primero MLP chico/mediano/grande.
- Luego probar Conv1D-CVAE.
- Usar GRU/LSTM si hay tiempo.
- Transformer-CVAE solo si se quiere mostrar una variante muy avanzada, porque
  con 24 puntos puede ser mas complejo de lo necesario.

## Bloque 5: regularizacion variacional

Estas variantes atacan directamente la calidad generativa y el uso del espacio
latente.

| Variante | Descripcion | Que mirar |
|---|---|---|
| Beta sweep | Probar `0.001`, `0.005`, `0.01`, `0.05`, `0.1`, `0.5`, `1` | Trade-off reconstruccion, KL y diversidad |
| Warm-up sweep | Warm-up de 25, 50, 100 epocas | Estabilidad del KL |
| Cyclical KL annealing | Beta sube y baja en ciclos | Evitar posterior collapse |
| Free bits | Minimo KL por dimension latente | Forzar uso de `z` |
| Conditional prior CVAE | Aprender `p(z|C)` en vez de usar `N(0,I)` | Variabilidad dependiente de condiciones |
| Mixture prior CVAE | Prior como mezcla de gaussianas | Capturar multiples modos de perfil |
| Hierarchical CVAE | Latente para nivel y latente para forma | Separar factores de variacion |

Experimento clave recomendado:

- Comparar baseline actual contra Conditional Prior CVAE.

Motivo:

- En el baseline se genera con `z ~ N(0,I)` para todos los contextos.
- Pero la variabilidad de un dia habil de verano no deberia ser igual a la de
  un domingo templado.
- Aprender `p(z|C)` es una mejora metodologicamente fuerte y facil de justificar.

## Bloque 6: funciones de perdida especificas del dominio

El baseline usa MSE horario + KL. Para demanda electrica conviene agregar
objetivos que reflejen lo que importa en el sistema.

| Loss | Formula conceptual | Objetivo |
|---|---|---|
| MSE horario | Error punto a punto | Reconstruccion general |
| Energy loss | Error en suma diaria | Controlar nivel diario |
| Peak loss | Error en demanda maxima | Controlar picos |
| Ramp loss | Error en diferencias hora a hora | Controlar dinamica |
| Shape loss | Error entre perfiles normalizados | Controlar forma relativa |
| Weighted hourly loss | Mas peso en horas pico o criticas | Mejorar periodos relevantes |
| Quantile / pinball loss | Error asimetrico por cuantiles | Modelar incertidumbre |

Versiones concretas:

1. `MSE + KL`.
2. `MSE + KL + energy_loss`.
3. `MSE + KL + energy_loss + ramp_loss`.
4. `MSE + KL + energy_loss + peak_loss + ramp_loss`.
5. `MSE + KL + shape_loss`, para variantes normalizadas.

Metricas a mirar:

- Si energy loss baja MAPE de energia diaria.
- Si peak loss baja peak MAE.
- Si ramp loss baja ramp MAE y evita curvas demasiado suaves.
- Si shape loss mejora perfiles por tipo de dia.

## Bloque 7: modelos generativos alternativos

Estos modelos sirven para mostrar que el trabajo no se limita a CVAE.

| Modelo | Descripcion | Ventaja | Riesgo |
|---|---|---|---|
| Conditional GAN | Genera perfil desde `C + ruido` y un discriminador evalua realismo | Puede producir curvas mas realistas | Entrenamiento inestable |
| WGAN-GP | GAN con distancia Wasserstein y gradient penalty | Mas estable que GAN clasica | Mas hiperparametros |
| Conditional Normalizing Flow | Aprende distribucion exacta `p(X|C)` | Probabilistico y avanzado | Implementacion mas compleja |
| Conditional Diffusion Model | Aprende a denoising de perfiles condicionado por `C` | Muy moderno y potente | Puede ser excesivo para pocos datos |
| Mixture Density Network | Predice mezcla de gaussianas condicionada por `C` | Probabilistico y mas simple que diffusion | Puede asumir independencia horaria si no se disena bien |
| Quantile Regression Forest | Predice cuantiles por hora | Clasico, robusto, interpretable | No genera trayectoria conjunta naturalmente |
| Gradient Boosting Quantile | Predice P10/P50/P90 por hora | Buen baseline probabilistico tabular | Dependencia horaria limitada |
| Gaussian Process | Regresion probabilistica con incertidumbre | Teoricamente elegante | Escala y extension multi-output |
| Temporal Fusion Transformer | Forecasting probabilistico con covariables | Muy avanzado para series temporales | Requiere reformular el problema |

Modelos generativos mas recomendables si hay tiempo:

1. WGAN-GP condicional.
2. Conditional Normalizing Flow.
3. Conditional Diffusion para perfiles de 24 horas.

## Bloque 8: modelos supervisados probabilisticos

Aunque el objetivo sea generativo, conviene incluir modelos probabilisticos no
necesariamente generativos.

| Modelo | Output | Uso |
|---|---|---|
| MLP deterministico | Media esperada de 24 horas | Baseline deep |
| MLP quantile | P10, P50, P90 por hora | Intervalos |
| Random Forest multi-output | 24 horas | Baseline clasico |
| XGBoost multi-output | 24 horas | Baseline tabular fuerte |
| XGBoost quantile por hora | Cuantiles horarios | Incertidumbre |
| Mixture Density Network | Distribucion por hora o conjunta | Probabilistico |

Preguntas:

- El CVAE genera mejor diversidad que un MLP deterministico?
- Los modelos tabulares predicen mejor energia diaria?
- Los cuantiles cubren mejor la variabilidad que el CVAE?

## Orden recomendado de implementacion

### Etapa 1: comparacion minima solida

1. Promedio por tipo de dia + mes.
2. KNN de dias similares.
3. MLP deterministico.
4. Random Forest o XGBoost multi-output.
5. CVAE C-full baseline actual.

Objetivo:

- Probar que el CVAE compite contra modelos simples y clasicos.

### Etapa 2: ablations del CVAE

1. CVAE C-minimal.
2. CVAE calendar-only.
3. CVAE weather-only.
4. CVAE sin tipo_dia.
5. CVAE con eventos.

Objetivo:

- Demostrar que se estudio el rol de cada grupo de variables.

### Etapa 3: mejoras generativas del CVAE

1. Beta sweep.
2. Warm-up sweep.
3. Free bits.
4. Conditional prior CVAE.
5. Loss con energia + rampas.

Objetivo:

- Mejorar diversidad y realismo sin perder reconstruccion.

### Etapa 4: representacion avanzada

1. CVAE normalized shape.
2. Modelo nivel + CVAE shape.
3. CVAE residual.
4. CVAE PCA.

Objetivo:

- Aprovechar la separacion nivel-forma observada en el EDA.

### Etapa 5: modelos avanzados extra

1. Conv1D-CVAE.
2. WGAN-GP condicional.
3. Conditional Normalizing Flow o Conditional Diffusion.

Objetivo:

- Mostrar comparacion con familias generativas modernas.

## Ranking practico si el tiempo es limitado

Si no se pueden implementar todos, priorizar:

1. Promedio por tipo de dia + mes.
2. KNN de dias similares.
3. Random Forest o XGBoost multi-output.
4. MLP deterministico.
5. CVAE C-full baseline.
6. CVAE C-minimal.
7. CVAE level-shape.
8. CVAE con energy/ramp loss.
9. Conv1D-CVAE.
10. Conditional prior CVAE.
11. WGAN-GP o Conditional Diffusion, solo si queda tiempo.

## Estructura sugerida para el informe

### 1. Baselines y motivacion

Presentar primero modelos simples y supervisados.

Mensaje:

> Antes de usar modelos generativos, se establecen referencias interpretables y
> modelos predictivos clasicos para cuantificar el valor agregado del CVAE.

### 2. CVAE baseline final

Presentar el baseline actual:

- C-full.
- Split temporal.
- Escalado train-only.
- Multi-seed.
- Early stopping.
- Beta warm-up.
- Metricas generativas.

Mensaje:

> El baseline CVAE ya no presenta colapso del KL y genera perfiles promedio
> plausibles por tipo de dia, aunque subestima levemente la variabilidad.

### 3. Ablations de condicionamiento

Comparar C-full, C-minimal, calendar-only y weather-only.

Mensaje:

> Se evalua empiricamente que informacion aporta cada grupo de variables.

### 4. Mejoras de arquitectura y perdida

Comparar representaciones, losses y arquitecturas.

Mensaje:

> Las variantes no solo cambian hiperparametros, sino que incorporan
> conocimiento del dominio electrico: energia, picos, rampas y forma.

### 5. Modelos generativos alternativos

Comparar con WGAN, flow o diffusion si se implementan.

Mensaje:

> El CVAE se ubica dentro de un conjunto mas amplio de modelos generativos
> condicionales, evaluados con el mismo protocolo.

## Tabla final recomendada

La tabla final del informe deberia tener una fila por modelo y columnas como:

- Familia de modelo.
- Condiciones usadas.
- Representacion de salida.
- Seeds.
- MAE horario test.
- Energy MAPE test.
- Peak MAE test.
- Peak hour accuracy.
- Ramp MAE.
- Group mean profile MAE.
- Diversity ratio.
- Observaciones cualitativas.

Ejemplo:

| Modelo | C | Salida | MAE | Energy MAPE | Peak MAE | Diversity ratio | Comentario |
|---|---|---|---:|---:|---:|---:|---|
| Promedio tipo dia + mes | calendario | perfil promedio | ... | ... | ... | ... | baseline interpretable |
| XGBoost | C-full | 24 horas | ... | ... | ... | ... | fuerte en nivel |
| MLP | C-full | 24 horas | ... | ... | ... | ... | deep deterministico |
| CVAE C-full | C-full | generativo | ... | ... | ... | ... | baseline generativo |
| CVAE level-shape | C-full | nivel + forma | ... | ... | ... | ... | separa variacion |
| Conv1D-CVAE | C-full | generativo | ... | ... | ... | ... | mejora dinamica |
| Conditional prior CVAE | C-full | generativo | ... | ... | ... | ... | prior depende de C |

## Narrativa metodologica central

La narrativa recomendada para defender el trabajo es:

> El trabajo no se limita a entrenar un unico CVAE. Primero se construyen
> baselines simples, modelos clasicos y modelos deterministas para establecer
> referencias. Luego se desarrolla un CVAE baseline corregido metodologicamente.
> A partir de ese baseline se realizan ablations de condicionamiento,
> representacion, regularizacion variacional, arquitectura y funcion de perdida.
> Finalmente, se comparan modelos generativos alternativos usando el mismo
> protocolo temporal y las mismas metricas de realismo, diversidad, picos,
> rampas y consistencia por tipo de dia.

## Criterio de exito

Un modelo se considera mejor que el baseline si:

- Mantiene o mejora MAE/RMSE horario.
- Mejora Energy MAPE.
- Mejora Peak MAE o peak hour accuracy.
- Reduce Ramp MAE.
- Mantiene el KL lejos de cero, si es variacional.
- Genera perfiles medios por tipo de dia plausibles.
- Mantiene diversity ratio cercano a 1.
- No empeora especialmente feriados o dias con pocos ejemplos.

## Riesgos a mencionar

- El test 2025-2026 puede tener patrones nuevos no vistos en train.
- Feriados tienen pocos casos, por lo que sus metricas son ruidosas.
- Modelos generativos pueden producir buenos promedios pero malos perfiles
  individuales.
- Modelos deterministas suelen tener mejor MAE pero menor diversidad.
- GAN, flow y diffusion pueden ser mas dificiles de entrenar con pocos datos.
- Clima horario podria mejorar, pero aumenta dimensionalidad y complejidad.

## Conclusion operativa

La ruta mas fuerte para un trabajo avanzado pero realista es:

1. Consolidar baselines simples y modelos supervisados.
2. Comparar CVAE C-full contra C-minimal y ablations.
3. Implementar CVAE level-shape.
4. Incorporar energy loss y ramp loss.
5. Probar Conv1D-CVAE.
6. Probar Conditional Prior CVAE.
7. Si queda tiempo, agregar un modelo generativo alternativo moderno.

Con esta planificacion, el proyecto muestra una metodologia completa: control
de leakage, validacion temporal, comparacion multi-modelo, ablations,
regularizacion generativa, conocimiento del dominio y evaluacion estadistica de
perfiles sinteticos.
