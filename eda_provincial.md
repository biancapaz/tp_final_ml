# EDA Provincial — Demanda Eléctrica por Provincia

## Relación con el EDA Nacional

Este EDA complementa al EDA nacional (`eda_organizacion.md`). Todo lo que está en el EDA nacional aplica también aquí — calidad del dataset, análisis de demanda diaria, forma vs nivel, calendario, clima, outliers y eventos sociales. No repetir esos análisis: referenciarlos y agregar solo lo que es nuevo por tener granularidad provincial.

La diferencia central es que ahora la unidad de análisis es el sistema completo de 22 provincias:

```text
fecha | buenos_aires_01 | buenos_aires_02 | ... | tucuman_24
```

Eso abre análisis de distribución espacial, correlaciones entre provincias y heterogeneidad que no eran posibles con el dato nacional agregado.

---

## 1. Calidad del Dataset Provincial

Además de los chequeos del EDA nacional, verificar lo específico de tener 22 series:

- Confirmar que las 22 provincias están presentes en todos los días.
- Detectar provincias con valores nulos o cero sistemático en ciertos períodos.
- Verificar que la suma de provincias coincide con el TOTAL del archivo original — si hay diferencia, cuantificarla.
- Revisar si alguna provincia tiene saltos abruptos que puedan indicar cambio de fuente o error de medición.
- Comparar la suma provincial vs la serie nacional que ya teníamos — deberían coincidir.

### Preguntas a responder

- Hay provincias con datos faltantes en ciertos períodos?
- La suma de provincias coincide con el total nacional?
- Hay alguna provincia con comportamiento claramente anómalo que requiera tratamiento especial?

---

## 2. Distribución Espacial de la Demanda

Este es el análisis más nuevo respecto al EDA nacional. El objetivo es entender cómo se distribuye el consumo entre provincias y si esa distribución es estable o cambia.

### Participación provincial

- Participación promedio de cada provincia en la demanda nacional (%).
- Ranking de provincias por consumo promedio.
- Participación provincial por estación del año — ¿cambia en verano vs invierno?
- Participación provincial por hora del día — ¿hay horas donde Buenos Aires domina más?

### Visualizaciones recomendadas

- Barplot de participación promedio por provincia.
- Heatmap provincia × hora del día (participación promedio).
- Heatmap provincia × mes (participación promedio).
- Mapa de Argentina coloreado por consumo promedio por provincia.
- Stacked area chart de demanda horaria apilada por provincia para un día típico.

### Preguntas a responder

- Qué porcentaje del sistema concentran Buenos Aires + Córdoba + Santa Fe?
- La distribución espacial cambia entre verano e invierno?
- Hay provincias cuya participación sube fuertemente en ciertos meses?
- La distribución espacial es más estable que el nivel total?

---

## 3. Shares Provinciales

Además de los MWh absolutos, calcular el share de cada provincia hora a hora:

```python
share_provincia_hora = demanda_provincia_hora / demanda_total_hora
```

Este es el vector normalizado que podría usarse como `x` alternativo en el experimento de shares del CVAE.

### Análisis recomendados

- Distribución de shares por provincia — ¿cuánto varían?
- Share promedio vs share en días extremos (ola de calor, ola de frío).
- ¿Qué provincias tienen shares más variables y cuáles más estables?
- Comparar perfiles de MWh absolutos vs perfiles de shares — ¿cambia la forma?

### Preguntas a responder

- Los shares son más estables que los MWh absolutos?
- Hay provincias cuyo share cambia mucho según el clima?
- Modelar shares en lugar de MWh facilitaría el aprendizaje del CVAE?

---

## 4. Correlaciones Entre Provincias

El CVAE conjunto aprende correlaciones entre provincias. El EDA debe mostrar si esas correlaciones existen y son estructurales.

### Análisis recomendados

- Matriz de correlación entre perfiles diarios de las 22 provincias.
- ¿Hay grupos de provincias que se mueven juntas?
- Correlación entre provincias vecinas geográficamente.
- Correlación entre provincias con perfil económico similar.
- Correlación en días hábiles vs fines de semana — ¿cambia la estructura?
- Correlación en verano vs invierno.

### Visualizaciones recomendadas

- Heatmap de correlación provincia × provincia.
- Clustermap (correlación + clustering jerárquico) para identificar grupos naturales.
- Scatter de demanda Buenos Aires vs demanda Córdoba coloreado por mes.

### Preguntas a responder

- Cuando sube Buenos Aires, suben todas las demás o hay excepciones?
- Hay provincias que se mueven en sentido contrario en algún contexto?
- Los grupos de correlación coinciden con regiones geográficas o con tipo de economía?
- La estructura de correlación justifica modelar el sistema completo en lugar de provincias independientes?

---

## 5. Heterogeneidad Provincial

Las provincias tienen tamaños muy distintos. Eso afecta cómo el modelo las va a aprender.

### Análisis recomendados

- Coeficiente de variación por provincia — ¿quién es más volátil?
- Distribución de energía diaria por provincia — comparar escala y dispersión.
- Perfil horario promedio por provincia superpuestos en el mismo gráfico.
- Ratio max/min diario por provincia — ¿quién tiene más amplitud horaria?
- Comparar perfil de provincia grande (Buenos Aires) vs provincia pequeña (Santa Cruz).

### Visualizaciones recomendadas

- Boxplot de energía diaria por provincia ordenado por mediana.
- Grilla de perfiles horarios promedio, uno por provincia.
- Scatter de coeficiente de variación vs consumo promedio — ¿las pequeñas son más ruidosas?

### Preguntas a responder

- Las provincias pequeñas tienen perfiles más ruidosos y menos predecibles?
- La forma horaria es similar en todas las provincias o hay formas distintas?
- Hay provincias con pico a una hora distinta al resto?
- El modelo va a tener más dificultad con provincias pequeñas?

---

## 6. Relación Indicadores Socioeconómicos → Demanda

Este análisis valida que los indicadores que van a entrar en `c` efectivamente explican diferencias entre provincias.

### Análisis recomendados

- Scatter de consumo promedio por provincia vs población.
- Scatter de consumo promedio por provincia vs PBI per cápita.
- Scatter de consumo promedio por provincia vs participación industrial.
- ¿La relación es lineal o hay outliers (Neuquén por energía, Patagonia por clima)?
- Correlación entre indicadores entre sí — ¿hay multicolinealidad?

### Visualizaciones recomendadas

- Scatter consumo vs población etiquetado por provincia.
- Scatter consumo vs PBI per cápita etiquetado por provincia.
- Heatmap de correlación entre indicadores socioeconómicos.

### Preguntas a responder

- La población explica bien el consumo provincial?
- El PBI per cápita agrega información más allá de la población?
- Hay provincias que consumen mucho más o menos de lo esperado por sus indicadores?
- Los indicadores elegidos son suficientes o falta alguno?

---

## 7. Clima por Provincia

A diferencia del EDA nacional donde había un clima efectivo único, ahora cada provincia tiene su propio clima. El análisis debe mostrar si esa heterogeneidad climática explica heterogeneidad en la demanda.

### Análisis recomendados

- Temperatura media por provincia y mes — mostrar la diversidad climática del país.
- Días de calor extremo (grado_calor > 0) por provincia.
- Días de frío extremo (grado_frio > 0) por provincia.
- Correlación temperatura → demanda por provincia — ¿es igual en todas?
- ¿Provincias del norte tienen más efecto de calor, provincias del sur más de frío?
- Comparar la relación temperatura-demanda en Buenos Aires vs Neuquén vs Formosa.

### Visualizaciones recomendadas

- Heatmap provincia × mes de temperatura media.
- Scatter temperatura vs demanda por provincia en subplots.
- Mapa de Argentina coloreado por temperatura media anual.

### Preguntas a responder

- La relación temperatura-demanda varía según la provincia?
- Hay provincias donde el frío importa más que el calor o viceversa?
- El clima provincial explica variabilidad que el clima nacional agregado no capturaba?

---

## 8. PCA sobre el Vector x Provincial

El PCA sobre el vector completo de 528 dimensiones es más rico que el PCA nacional porque tiene estructura espacial además de temporal.

### Análisis recomendados

- Varianza explicada por los primeros componentes.
- Interpretar qué captura cada componente:
  - ¿PC1 es nivel total nacional?
  - ¿PC2 es distribución norte-sur?
  - ¿PC3 es asimetría mañana-tarde?
- Visualizar días en espacio PCA 2D coloreado por:
  - mes;
  - día de semana;
  - temperatura media nacional;
  - provincia con mayor demanda del día;
  - año.
- Visualizar los loadings de PC1 y PC2 — ¿qué provincia-hora tiene más peso?

### Preguntas a responder

- Cuántos componentes capturan el 90% de la varianza?
- El primer componente es nivel general o distribución espacial?
- Los días se agrupan por estación, por día de semana, o por otra cosa?
- El espacio latente tiene estructura interpretable antes de entrenar el modelo?

---

## 9. Resultado Esperado del EDA Provincial

Al finalizar este EDA, deberíamos poder responder:

- Qué provincias dominan el sistema y cuánto varía esa dominancia?
- Las provincias están fuertemente correlacionadas o tienen dinámicas independientes?
- Tiene sentido modelar el sistema completo o hay provincias que se comportan de forma muy distinta?
- Los indicadores socioeconómicos elegidos explican las diferencias entre provincias?
- El clima provincial agrega información útil respecto al clima nacional?
- Qué tan ruidosas son las provincias pequeñas y qué implica eso para el modelo?
- Modelar shares tiene sentido o los MWh absolutos son suficientes?
- Cuántos componentes del PCA capturan la estructura del sistema?

Estas respuestas informan directamente las decisiones de modelado: arquitectura del CVAE, features de `c`, y métricas de evaluación por provincia.
