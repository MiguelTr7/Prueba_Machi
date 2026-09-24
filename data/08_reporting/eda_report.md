# Analisis Exploratorio de Datos (EDA)

> Documento generado por el pipeline (`analisis_exploratorio_node`).
> Se regenera con `kedro run`; no se edita a mano.

## 1. Bitacora de vuelos — estructura y calidad

**11,074,197 filas · 16 columnas** (1999-12-31 a 2026-07-31)

| columna | tipo | nulos | % nulos | valores_distintos | ejemplo |
| --- | --- | --- | --- | --- | --- |
| dt_operacion | datetime64[us, America/Santiago] | 0 | 0.0 | 6479049 | 1999-12-31 21:00:00-03:00 |
| aeropuerto_oaci | object | 0 | 0.0 | 69 | SCEL |
| tipo_operacion | object | 0 | 0.0 | 4 | A |
| aeropuerto_dgac_orig_dest | object | 0 | 0.0 | 1668 | SPIM |
| aerolinea_dgac | object | 0 | 0.0 | 2555 | LAN |
| numero_vuelo | object | 0 | 0.0 | 7424 | 631 |
| actividad_cod | object | 0 | 0.0 | 26 | U |
| matricula | object | 0 | 0.0 | 13775 | CCCDM |
| modelo_avion | object | 0 | 0.0 | 1283 | B763 |
| modelo_avion_desc | object | 0 | 0.0 | 2322 | 767-352ER |
| es_internacional | bool | 0 | 0.0 | 2 | True |
| pmd | float64 | 0 | 0.0 | 786 | 187.0 |
| pmd_fue_imputado | int8 | 0 | 0.0 | 2 | 0 |
| mes_id | int64 | 0 | 0.0 | 320 | 199912 |
| internacional_domestico | object | 0 | 0.0 | 2 | I |
| cnt_operaciones | float64 | 400 | 0.0 | 2826 | 3460.0 |

### Resumen estadistico de las variables numericas

| variable | count | mean | std | min | 1% | 25% | 50% | 75% | 95% | 99% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pmd | 11074197.0 | 45.12 | 61.33 | 0.1 | 0.6 | 1.16 | 7.0 | 77.0 | 185.0 | 276.0 | 2100.0 |
| mes_id | 11074197.0 | 201416.86 | 766.12 | 199912.0 | 200004.0 | 200803.0 | 201502.0 | 202107.0 | 202508.0 | 202605.0 | 202607.0 |
| cnt_operaciones | 11073797.0 | 2810.05 | 2413.08 | 1.0 | 61.0 | 962.0 | 1718.0 | 4401.0 | 7572.0 | 8970.0 | 10722.0 |

## 2. Duplicados

| tipo | cantidad | % del total |
| --- | --- | --- |
| Filas identicas en todas sus columnas | 1,122 | 0.01 |
| Repeticiones de (fecha-hora, aeropuerto, matricula, tipo) | 2,470 | 0.022 |

Los duplicados exactos son operaciones que la JAC registra mas de una vez. **No se eliminan**: sin un identificador unico de operacion no se puede distinguir un registro repetido de dos movimientos reales muy seguidos, y borrarlos a ciegas sesgaria los conteos por aeropuerto.

## 3. Valores extremos del PMD

El PMD (Peso Maximo de Despegue, en toneladas) es la variable numerica con la cola mas larga del dataset.

| variable | q1 | q3 | iqr | limite_inferior | limite_superior | outliers_inferiores | outliers_superiores | pct_outliers | maximo | minimo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pmd | 1.16 | 77.0 | 75.84 | -112.6 | 190.76 | 0 | 336096 | 3.03 | 2100.0 | 0.1 |

El 3% de los vuelos cae fuera del rango intercuartil. **Mirar quien es cada uno separa tres situaciones distintas**, y solo una de ellas es normal.

### (a) Cola alta legitima — aeronaves de fuselaje ancho

73,646 vuelos entre 300 y 650 toneladas. Son modelos reales (B747, B777, A340) y el AN-225, que efectivamente pesa 640 t. **Se conservan**: transformarlos con logaritmo basta para que no dominen la escala al modelar.

| modelo_avion | operaciones | pmd_max |
| --- | --- | --- |
| B744 | 22102 | 448.0 |
| B773 | 14187 | 352.0 |
| A346 | 10244 | 380.0 |
| B77L | 6548 | 387.0 |
| B748 | 6256 | 448.0 |

### (b) Error de unidad — kilogramos registrados como toneladas

**1,078 vuelos superan las 650 toneladas**, mas que cualquier aeronave que haya volado. Al mirar el modelo queda claro que no son aviones gigantes sino **aviones pequeños con el peso en kilogramos**: un T-34 Mentor pesa 1.3 toneladas y aparece con 1 340; un helicoptero R-44 pesa 1.1 t y figura con 1 000.

| modelo_avion | modelo_avion_desc | operaciones | pmd_registrado |
| --- | --- | --- | --- |
| T34T | A-45 | 1007 | 1340.0 |
| AL3 | AL3 | 28 | 2100.0 |
| C210 | 210 | 20 | 1500.0 |
| R44 | R-44II | 15 | 1000.0 |
| B38M | 737 MAX 8 | 4 | 710.0 |
| C206 | T206H | 2 | 1640.0 |

**No se corrigen automaticamente.** Dividir por 1 000 arreglaria estos casos, pero exigiria decidir por umbral cuales convertir, y un umbral mal puesto danaria registros correctos. Son el 0.01% de los datos y ninguno entra en la PoC de El Tepual, asi que se documentan como anomalia conocida en vez de parcharse a ciegas.

### (c) Ceros — nulos disfrazados

Un avion no puede pesar cero. En la fuente cruda hay **25,821 registros con `pmd = 0`**

La primera version del pipeline imputaba solo los `NaN`, asi que estos pasaban intactos al modelo como aviones sin peso. **Ahora se tratan como faltantes** y entran a la imputacion por mediana del modelo de avion, igual que el resto: en total se imputa el PMD de **300,936 vuelos (2.72%)**.

> Ninguna de estas anomalias afecta a la PoC: de los 1,078 registros con unidad erronea, solo 4 ocurren en El Tepual y ninguno es vuelo regular del periodo 2020-2026.

## 4. Clima diario de Puerto Montt (estacion 85799)

**2,192 dias** (2020-01-01 a 2025-12-31), sin dias faltantes.

| columna | tipo | nulos | % nulos | valores_distintos | ejemplo |
| --- | --- | --- | --- | --- | --- |
| fecha_cruce | datetime64[ns] | 0 | 0.0 | 2192 | 2020-01-01 00:00:00 |
| tavg | float64 | 0 | 0.0 | 183 | 15.7 |
| tmin | float64 | 0 | 0.0 | 188 | 8.0 |
| tmax | float64 | 0 | 0.0 | 203 | 22.0 |
| amplitud_termica | float64 | 0 | 0.0 | 375 | 14.0 |
| prcp | float64 | 0 | 0.0 | 206 | 0.0 |
| dia_lluvioso | int8 | 0 | 0.0 | 2 | 0 |
| wspd | float64 | 0 | 0.0 | 347 | 13.7 |
| wdir | float64 | 9 | 0.41 | 312 | 188.0 |
| viento_cruzado | float64 | 0 | 0.0 | 2078 | 4.2335328229367795 |
| viento_frontal | float64 | 0 | 0.0 | 2095 | 13.029474273243602 |
| pres | float64 | 0 | 0.0 | 298 | 1013.8 |

### Resumen estadistico del clima

| variable | count | mean | std | min | 1% | 25% | 50% | 75% | 95% | 99% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tavg | 2192.0 | 10.29 | 3.73 | -1.3 | 1.7 | 7.7 | 10.4 | 13.1 | 16.1 | 17.81 | 19.4 |
| tmin | 2192.0 | 6.02 | 4.02 | -8.1 | -3.0 | 3.2 | 6.1 | 9.0 | 12.0 | 13.9 | 16.6 |
| tmax | 2192.0 | 15.39 | 4.45 | 5.4 | 8.0 | 11.9 | 15.0 | 18.6 | 23.0 | 26.2 | 29.4 |
| amplitud_termica | 2192.0 | 9.36 | 3.81 | -0.2 | 2.3 | 6.4 | 9.0 | 11.92 | 16.0 | 19.41 | 24.7 |
| prcp | 2192.0 | 2.83 | 7.33 | 0.0 | 0.0 | 0.0 | 0.0 | 2.5 | 15.3 | 26.08 | 208.8 |
| wspd | 2192.0 | 14.23 | 8.01 | 1.4 | 3.8 | 8.8 | 12.2 | 17.5 | 30.09 | 43.82 | 55.6 |
| viento_cruzado | 2192.0 | 5.95 | 4.33 | 0.0 | 0.14 | 3.03 | 5.2 | 7.87 | 13.49 | 21.24 | 41.41 |
| pres | 2192.0 | 1017.43 | 5.58 | 992.5 | 1001.39 | 1014.4 | 1017.7 | 1021.0 | 1025.94 | 1029.52 | 1035.1 |

Llueve en el **44.9%** de los dias. Ese solo dato explica por que la lluvia no diferencia dias buenos de malos en El Tepual: es la condicion normal, no una anomalia.

## 5. Subconjunto de la PoC — El Tepual (SCTE)

**71,606 vuelos regulares** con horario reconstruido, 22 aerolineas y 263 numeros de vuelo distintos.

### Cobertura por año

| anio | vuelos | tasa_retraso | desvio_mediano |
| --- | --- | --- | --- |
| 2020 | 6168 | 0.14 | 0.0 |
| 2021 | 10243 | 0.149 | 0.0 |
| 2022 | 12458 | 0.133 | 0.0 |
| 2023 | 13198 | 0.165 | 0.0 |
| 2024 | 14850 | 0.177 | 0.0 |
| 2025 | 14689 | 0.138 | 0.0 |

El volumen de 2020 es **menos de la mitad** que el de 2024: el periodo arranca en plena pandemia. El modelo entrena con dos regimenes operativos distintos tratados como uno solo (ver la seccion de sesgos del README).

### Principales operadores

| aerolinea_dgac | vuelos | tasa_retraso | desvio_p75 |
| --- | --- | --- | --- |
| SKU | 24904 | 0.149 | 7.0 |
| LXP | 20607 | 0.141 | 6.5 |
| LAN | 17474 | 0.137 | 6.5 |
| JAT | 8209 | 0.212 | 11.0 |
| 6R | 130 | 0.292 | 32.75 |
| 1D | 58 | 0.293 | 23.125 |
| 30O | 56 | 0.286 | 28.625 |
| 60F | 41 | 0.415 | 55.0 |

## 6. El target de retraso

| vuelos | tasa de retraso (>15 min) | retraso severo (>60 min) | desvio p05 | desvio mediano | desvio p95 | desviacion estandar |
| --- | --- | --- | --- | --- | --- | --- |
| 71,606 | 15.2% | 4.2% | -29.0 | 0.0 | 52.0 | 56.8 |

La distribucion esta centrada en cero, con cola izquierda corta y cola derecha larga. **Esa asimetria es la firma de un retraso real**: los vuelos se atrasan mucho y se adelantan poco. Si fuera ruido de medicion, seria simetrica.

## Figuras generadas

- `images/eda_08_outliers_pmd.png` — distribucion y valores extremos del PMD
- `images/eda_09_target_retraso.png` — el target por año, hora y operador
