# Organizacion del EDA

## Objetivo del EDA

El objetivo del EDA no es entrenar el modelo, sino entender la distribucion real de los perfiles diarios de demanda electrica. Como el proyecto busca generar perfiles horarios realistas, el EDA sirve para decidir:

- que variables usar como condicion del modelo;
- si conviene modelar dias completos o filas horarias;
- que patrones debe reproducir un modelo generativo;
- que dias son normales, atipicos o explicables por eventos externos;
- que metricas y visualizaciones usar para comparar datos reales vs datos generados.

La unidad principal de analisis deberia ser el dia completo, representado por las 24 demandas horarias:

```text
fecha | demanda_01 | demanda_02 | ... | demanda_24
```

Las variables de calendario y clima se usan como contexto del dia.

---

## 1. Calidad del Dataset

Antes de analizar patrones, hay que confirmar que la base este completa y sea consistente.

### Chequeos basicos

- Verificar el rango temporal real del dataset.
- Confirmar que cada dia tenga exactamente 24 horas.
- Detectar fechas faltantes.
- Detectar horas duplicadas.
- Detectar dias incompletos.
- Confirmar que la variable `hora` va de 1 a 24.
- Revisar valores nulos en demanda, calendario y clima.
- Revisar valores imposibles:
  - demanda negativa;
  - temperatura extrema irreal;
  - humedad fuera del rango 0-100;
  - viento negativo.

### Chequeos especificos del dataset

- Verificar el empalme entre los dos Excel originales.
- Revisar el cambio entre datos regionales y datos provinciales.
- Confirmar que 2024-02-29 este incluido correctamente.
- Confirmar que la suma regional/provincial coincida con la columna `TOTAL` en los archivos originales.
- Revisar si 2026 debe tratarse aparte porque esta incompleto, ya que solo llega hasta abril.

### Preguntas a responder

- Hay dias faltantes o incompletos?
- Hay valores raros que parezcan errores de medicion?
- El empalme entre fuentes introduce un salto artificial?
- El periodo 2026 se puede usar igual que el resto o requiere cuidado?

---

## 2. Demanda Diaria

Este bloque busca entender como se comporta la demanda electrica en terminos generales.

### Variables derivadas por dia

A partir de las 24 demandas horarias, conviene calcular:

- energia diaria total;
- demanda media diaria;
- demanda maxima diaria;
- demanda minima diaria;
- hora del pico diario;
- hora del valle diario;
- amplitud diaria: maximo menos minimo;
- factor de carga: demanda media dividida por demanda maxima;
- rampas horarias: diferencia entre una hora y la anterior.

### Analisis de perfiles

- Perfil horario promedio general.
- Perfil horario promedio por anio.
- Perfil horario promedio por mes.
- Perfil horario promedio por estacion:
  - verano;
  - otonio;
  - invierno;
  - primavera.
- Perfil horario promedio por dia de semana.
- Perfil horario promedio para:
  - dias habiles;
  - sabados;
  - domingos;
  - feriados.

### Distribuciones

- Distribucion de demanda por hora.
- Distribucion de energia diaria total.
- Distribucion de demanda maxima diaria.
- Distribucion de demanda minima diaria.
- Distribucion de hora del pico.
- Distribucion de amplitud diaria.
- Distribucion de rampas horarias.

### Preguntas a responder

- Cual es el perfil diario promedio?
- La forma del perfil cambia mucho segun el mes?
- Los fines de semana tienen una forma distinta?
- En que horas suelen aparecer los picos?
- Hay dias con picos o caidas especialmente raras?

---

## 3. Forma vs Nivel

En un modelo generativo de perfiles diarios, es importante separar dos conceptos:

- nivel: cuanto se consume en total;
- forma: como se distribuye ese consumo durante las 24 horas.

Dos dias pueden tener el mismo total diario pero formas distintas. Tambien pueden tener formas parecidas pero distinto nivel de consumo.

### Analisis recomendados

- Analizar perfiles en MWh reales.
- Analizar perfiles normalizados por energia diaria.
- Comparar curvas diarias con mismo total y distinta forma.
- Comparar curvas con forma similar y distinto nivel.
- Ver si la variabilidad principal esta en el nivel o en la forma.
- Evaluar si PCA o autoencoder capturan principalmente nivel total, forma horaria o ambas cosas.

### Preguntas a responder

- El modelo deberia aprender principalmente nivel, forma o ambas cosas?
- Los primeros componentes de PCA representan escala general o cambios de forma?
- Tiene sentido normalizar perfiles para algun experimento?

---

## 4. Calendario

La demanda electrica suele depender mucho de las rutinas sociales y laborales. Por eso, el calendario es una parte clave del analisis y probablemente tambien del vector condicional `C`.

### Variables calendario

- Mes.
- Dia de semana.
- Fin de semana.
- Feriado.
- Vispera de feriado.
- Dia posterior a feriado.
- Estacion del anio.

### Analisis recomendados

- Demanda por mes.
- Demanda por dia de semana.
- Demanda en dias habiles vs fines de semana.
- Demanda en sabados vs domingos.
- Demanda en feriados vs dias habiles comparables.
- Demanda en feriados que caen lunes o viernes.
- Demanda en fines de semana largos.
- Demanda en visperas de feriado.
- Demanda en dias posteriores a feriado.
- Demanda en fechas especiales:
  - 1 de enero;
  - 24 de diciembre;
  - 25 de diciembre;
  - 31 de diciembre;
  - 1 de mayo;
  - 9 de julio;
  - otros feriados nacionales importantes.

### Preguntas a responder

- Los feriados se parecen mas a domingos o tienen un patron propio?
- Los lunes y viernes se comportan distinto al resto de los dias habiles?
- Hay estacionalidad mensual fuerte?
- Enero y febrero se comportan distinto por vacaciones?
- El calendario explica cambios de forma o principalmente cambios de nivel?

---

## 5. Clima

El clima puede afectar fuertemente el consumo electrico, especialmente por calefaccion y aire acondicionado. Como se trata de Argentina, es esperable que haya patrones distintos entre verano e invierno.

### Variables climaticas a analizar

Desde el clima horario, conviene construir variables diarias como:

- temperatura media;
- temperatura minima;
- temperatura maxima;
- amplitud termica;
- sensacion termica media;
- sensacion termica minima;
- sensacion termica maxima;
- amplitud de sensacion termica;
- humedad media;
- viento medio.

Tambien se pueden construir variables derivadas:

- indicador de dia muy caluroso;
- indicador de dia muy frio;
- cooling degree days;
- heating degree days.

### Analisis recomendados

- Distribucion de temperatura efectiva.
- Distribucion de sensacion termica.
- Distribucion de humedad.
- Distribucion de viento.
- Clima por mes.
- Clima por estacion.
- Dias de calor extremo.
- Dias de frio extremo.
- Temperatura media diaria vs energia diaria total.
- Temperatura maxima diaria vs demanda maxima diaria.
- Sensacion termica maxima vs demanda maxima diaria.
- Humedad vs demanda, especialmente en verano.
- Viento vs demanda.
- Amplitud termica diaria vs forma del perfil.
- Temperatura horaria vs demanda horaria.

### Preguntas a responder

- La relacion entre temperatura y demanda tiene forma de U?
- El calor afecta mas que el frio?
- La temperatura maxima explica mejor el pico diario que la temperatura media?
- La sensacion termica explica mejor la demanda que la temperatura real?
- El efecto del clima cambia segun la hora del dia?
- Conviene usar clima diario agregado o clima horario como condicion?

---

## 6. Relaciones Principales

Estas son relaciones importantes para decidir que variables entran al modelo y para interpretar los patrones de consumo.

| Relacion | Que buscar |
|---|---|
| Demanda diaria total vs temperatura media | Posible curva en U: mas consumo con frio o calor extremos |
| Demanda maxima diaria vs temperatura maxima | Efecto de aire acondicionado en dias calurosos |
| Demanda horaria vs temperatura horaria | Efecto del clima durante el dia |
| Perfil diario vs mes | Estacionalidad de la forma |
| Perfil diario vs dia de semana | Rutinas laborales y sociales |
| Demanda vs feriado | Cambio de nivel o forma |
| Demanda vs domingo | Patron residencial distinto |
| Demanda vs humedad en verano | Posible relacion con sensacion termica |
| Demanda vs sensacion termica | Comparar contra temperatura real |
| Demanda vs anio | Tendencia estructural |
| Hora pico vs estacion | Cambios en horario de maxima demanda |
| Rampas vs estacion | Cambios bruscos por rutinas o clima |
| Componentes PCA vs calendario | Interpretar que capturan los componentes |
| Componentes PCA vs clima | Ver si el clima explica variabilidad de perfiles |
| Clusters de dias vs clima/calendario | Identificar tipos naturales de dias |

---

## 7. Eventos Sociales Argentinos

Como el dataset es real y cubre 2021-2026, hay eventos sociales que pueden haber afectado el consumo. Estos eventos no necesariamente deberian entrar al modelo desde el inicio, pero si deberian usarse para explicar anomalías.

### COVID y normalizacion posterior

El dataset empieza en 2021, por lo que no captura la cuarentena estricta inicial de 2020. Sin embargo, si puede capturar efectos de restricciones, reaperturas y cambios de movilidad durante 2021.

Analizar:

- abril-mayo 2021;
- aislamiento estricto del 22 al 31 de mayo de 2021;
- segunda ola de COVID;
- enero 2022 por la ola de Omicron;
- comparacion entre 2021 y anios posteriores.

Preguntas:

- 2021 tiene perfiles distintos al resto?
- Se observa una normalizacion progresiva?
- Cambian mas los dias habiles que los fines de semana?

### Mundial Qatar 2022

Marcar los partidos de Argentina y analizar el consumo durante:

- hora previa al partido;
- horario del partido;
- dos a cuatro horas posteriores;
- dia completo del partido.

Fechas clave de partidos de Argentina:

- 22/11/2022 07:00;
- 26/11/2022 16:00;
- 30/11/2022 16:00;
- 03/12/2022 16:00;
- 09/12/2022 16:00;
- 13/12/2022 16:00;
- 18/12/2022 12:00, final.

Tambien analizar:

- 20/12/2022, festejos y recibimiento de la seleccion.

Preguntas:

- Durante los partidos baja o sube la demanda?
- Hay cambio en la forma horaria del dia?
- La final tiene un comportamiento distinto al de otros domingos comparables?
- El dia de festejos aparece como outlier?

### Elecciones presidenciales 2023

Fechas a marcar:

- PASO: 13/08/2023;
- eleccion general: 22/10/2023;
- balotaje: 19/11/2023.

Analizar:

- manana y tarde del domingo electoral;
- noche electoral;
- dia posterior a la eleccion;
- comparacion contra domingos similares.

Preguntas:

- Los domingos electorales se comportan distinto a domingos normales?
- Hay cambios en la tarde por movilidad social?
- Hay cambios en la noche por seguimiento de resultados?

---

## 8. Como Analizar Eventos Sin Enganiarnos

Para eventos sociales, no conviene comparar contra el promedio general, porque eso puede confundir efectos de calendario, clima o estacion.

La comparacion deberia hacerse contra dias similares:

```text
mismo mes
mismo dia de semana
temperatura parecida
misma condicion de feriado/no feriado
mismo rango horario
```

Una estrategia util es estimar una demanda esperada segun clima y calendario, y luego analizar el residuo:

```text
residuo = demanda_observada - demanda_esperada_por_clima_y_calendario
```

Si el residuo es grande, el evento podria haber tenido un efecto real. Si no, tal vez el dia parecia raro solo por temperatura, estacionalidad o calendario.

---

## 9. Outliers y Dias Atipicos

No hay que borrar outliers automaticamente. Primero hay que entenderlos.

### Tipos de outliers

- Outliers por demanda horaria.
- Outliers por energia diaria total.
- Outliers por demanda maxima diaria.
- Outliers por demanda minima diaria.
- Outliers por hora del pico.
- Outliers por rampas horarias.
- Outliers por forma del perfil diario.
- Outliers en el espacio PCA.

### Clasificacion posible

- Explicables por clima extremo.
- Explicables por feriados.
- Explicables por eventos sociales.
- Explicables por cambios de fuente de datos.
- No explicables, posiblemente errores o eventos no registrados.

### Preguntas a responder

- Hay outliers que deberian conservarse porque representan variabilidad real?
- Hay outliers que parecen errores de datos?
- El modelo deberia aprender esos dias o conviene etiquetarlos aparte?

---

## 10. Visualizaciones Recomendadas

### Calidad y estructura

- Linea temporal de cobertura de datos.
- Conteo de horas por dia.
- Mapa de valores faltantes.
- Comparacion del empalme entre fuentes.

### Demanda

- Heatmap fecha x hora de demanda.
- Perfil horario promedio general.
- Curvas promedio con banda p25-p75.
- Boxplots por hora.
- Boxplots por mes.
- Boxplots por dia de semana.
- Calendar heatmap de energia diaria.
- Distribucion de energia diaria total.
- Distribucion de demanda maxima diaria.
- Distribucion de hora pico.

### Calendario

- Perfil promedio de dia habil vs sabado vs domingo vs feriado.
- Boxplot de energia diaria por dia de semana.
- Comparacion de feriados contra dias habiles similares.
- Comparacion de visperas y dias posteriores a feriado.

### Clima

- Scatter de demanda diaria total vs temperatura media.
- Scatter de demanda maxima diaria vs temperatura maxima.
- Scatter de demanda maxima diaria vs sensacion termica maxima.
- Curva suavizada de demanda vs temperatura.
- Perfil promedio para dias frios, templados y calurosos.
- Heatmap de demanda horaria segun temperatura horaria.

### Eventos

- Comparacion de dias de Mundial contra dias similares.
- Comparacion de domingos electorales contra domingos similares.
- Grafico horario del dia del evento vs promedio comparable.
- Residuos de demanda en fechas especiales.

### Representaciones para modelado

- PCA 2D de perfiles diarios coloreado por mes.
- PCA 2D coloreado por temperatura.
- PCA 2D coloreado por dia de semana.
- PCA 2D coloreado por feriado.
- PCA 2D coloreado por anio.
- Clusters de perfiles diarios.
- Perfil promedio de cada cluster.
- Distribucion de clusters por clima y calendario.

---

## 11. Preguntas Finales Que Deberia Responder el EDA

Al finalizar el EDA, deberiamos poder responder:

- La demanda esta mas explicada por calendario, clima o tendencia temporal?
- La forma diaria cambia mucho o cambia principalmente el nivel?
- Hay una tendencia creciente o decreciente entre 2021 y 2026?
- El anio 2021 se comporta distinto por pandemia y reapertura?
- Los feriados argentinos tienen un patron propio?
- Los domingos y feriados son parecidos?
- El calor afecta mas que el frio, o ambos importan?
- La temperatura afecta igual a todas las horas?
- La sensacion termica explica mejor la demanda que la temperatura?
- Hay eventos sociales que generen dias atipicos?
- Conviene excluir, conservar o etiquetar dias excepcionales?
- Que variables deberian entrar al vector condicional `C`?
- Que metricas deberian usarse despues para evaluar datos generados?

---

## 12. Resultado Esperado del EDA

El EDA deberia dejar como resultado:

- una descripcion clara del dataset;
- graficos que muestren patrones principales;
- una lista de variables utiles para el modelo condicional;
- una lista de eventos o dias atipicos relevantes;
- una decision sobre si modelar nivel, forma o ambos;
- criterios para evaluar si los perfiles generados son realistas.

La idea central es que el modelo generativo no solo produzca curvas suaves, sino curvas plausibles para el contexto argentino: dias habiles, fines de semana, feriados, estaciones, clima extremo y eventos sociales relevantes.
