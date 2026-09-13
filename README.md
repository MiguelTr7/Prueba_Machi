# Modelo ML — Operaciones Aeronauticas Chile (JAC)

> Proyecto de Machine Learning sobre la bitacora de vuelos de Chile (1999–2026).
> Fuentes: [datos.gob.cl — Junta de Aeronautica Civil](https://datos.gob.cl/dataset/operaciones-aeronaves) · [Meteostat — Puerto Montt, estacion 85799](https://meteostat.net/es/place/cl/puerto-montt?s=85799)

---

## Indice

**Definicion del proyecto**

- [¿De que trata este proyecto?](#de-que-trata-este-proyecto)
- [Problema de negocio y objetivos](#problema-de-negocio-y-objetivos)
- [KPIs — como se mide el exito](#kpis--como-se-mide-el-exito)
- [Metodologia — CRISP-DM](#metodologia--crisp-dm)
- [Fuentes de datos y herramientas](#fuentes-de-datos-y-herramientas)

**Desarrollo**

- [Arquitectura y Pipeline](#arquitectura-y-pipeline)
- [Como ejecutar el proyecto](#como-ejecutar-el-proyecto)
- [Preparacion de los datos](#preparacion-de-los-datos)
- [Analisis Exploratorio — Graficos clave](#analisis-exploratorio--graficos-clave)
- [Modelado No Supervisado — Clustering](#modelado-no-supervisado--clustering)
- [Modelado Supervisado — Clasificacion](#modelado-supervisado--clasificacion)

**Prediccion de retrasos (PoC Puerto Montt)**

- [Impacto de Negocio — por que predecir retrasos](#impacto-de-negocio--por-que-predecir-retrasos)
- [Como se construye el retraso si el dato no lo trae](#como-se-construye-el-retraso-si-el-dato-no-lo-trae)
- [El cruce con el clima](#el-cruce-con-el-clima)
- [¿Cuanto retrasa realmente el clima?](#cuanto-retrasa-realmente-el-clima)
- [El Score de Riesgo Operativo](#el-score-de-riesgo-operativo)
- [Ventana de Confianza — 7 a 14 dias](#ventana-de-confianza--7-a-14-dias)

**Cierre**

- [Etica, sesgos y privacidad](#etica-sesgos-y-privacidad)
- [Conclusiones y Decisiones de Negocio](#conclusiones-y-decisiones-de-negocio)
- [Trabajo futuro](#trabajo-futuro)
- [Estructura del proyecto](#estructura-del-proyecto)

---

## ¿De que trata este proyecto?

Chile tiene 69 aeropuertos activos y registra mas de **11 millones de operaciones aereas** entre 1999 y 2026. Esta base de datos — publicada abiertamente por la JAC — es una de las series historicas mas completas de transporte aereo en Latinoamerica.

El proyecto tiene **dos lineas de trabajo** que corren en paralelo:

**1. Modelo Base (baseline).** Demuestra que la arquitectura Kedro funciona de punta a punta: clasifica vuelos internacionales vs domesticos y agrupa aeropuertos por comportamiento operativo.

- ¿Como se distribuye el trafico aereo en Chile?
- ¿Se pueden agrupar los aeropuertos por su comportamiento operativo?
- ¿Se puede predecir si un vuelo es internacional solo con sus datos operativos?

**2. Modelo de Riesgo Operativo (`retrasos_ml`).** Ataca una pregunta con impacto economico directo, acotada a Puerto Montt como prueba de concepto:

- **¿Hasta que punto el clima puede retrasar un vuelo?**

La respuesta corta a esa ultima pregunta, adelantada aqui porque es el hallazgo central del proyecto: **mucho menos de lo que se supone**. El detalle esta en [¿Cuanto retrasa realmente el clima?](#cuanto-retrasa-realmente-el-clima).

---

## Problema de negocio y objetivos

### El problema

Un vuelo retrasado cuesta dinero, y cuesta en varios frentes a la vez: combustible quemado en tierra, horas de tripulacion fuera de itinerario, ocupacion de gate que bloquea a otra aeronave, compensaciones a pasajeros y perdida de conexiones. **Ninguno de esos costos aparece en los datos publicos** — pero los minutos si. Por eso el proyecto usa el retraso como **variable proxy del costo operativo**: no tenemos las facturas de la aerolinea, tenemos los minutos, y los minutos se convierten en pesos.

El problema concreto que se ataca: **un jefe de operaciones en El Tepual no sabe, con una o dos semanas de anticipacion, que dias van a ser malos.** Si lo supiera, podria mover tripulacion de reserva, ajustar el buffer entre rotaciones o avisar a los pasajeros antes de que lleguen al aeropuerto.

La hipotesis de partida — la que el equipo quiso poner a prueba — es que **el clima explica buena parte de esos dias malos**. Puerto Montt es un buen lugar para probarla: llueve el 44.9% de los dias.

### Objetivos

**Objetivo general.** Determinar si es posible anticipar el riesgo de retraso de un vuelo en El Tepual cruzando la bitacora operativa de la JAC con datos meteorologicos historicos, y cuantificar cuanto de ese riesgo explica efectivamente el clima.

**Objetivos especificos:**

| # | Objetivo | Estado |
|---|---|---|
| O1 | Construir un pipeline reproducible de ingesta, limpieza e integracion de las tres fuentes | ✅ Cumplido — 18 nodos Kedro, 146 s |
| O2 | Derivar una medida de retraso valida a partir de datos que **no** incluyen hora programada | ✅ Cumplido — horario reconstruido por slots |
| O3 | Cruzar vuelos y clima a nivel diario sin degradar el rendimiento | ✅ Cumplido — 71 606 vuelos, 0 sin clima |
| O4 | Cuantificar el efecto del clima sobre la tasa de retraso | ✅ Cumplido — el efecto es casi nulo salvo viento cruzado extremo |
| O5 | Entregar un Score de Riesgo accionable (0-100%) | ⚠️ **No alcanzado** — lift 1.12, insuficiente para decidir |
| O6 | Documentar sesgos, limites y consideraciones eticas del uso del modelo | ✅ Cumplido — ver [Etica, sesgos y privacidad](#etica-sesgos-y-privacidad) |

> **O5 no se alcanzo y se reporta como tal.** El modelo existe, corre y esta calibrado, pero no discrimina lo suficiente como para sostener una decision operativa. Presentarlo como exitoso seria el error mas grave que este proyecto podria cometer.

---

## KPIs — como se mide el exito

Los KPIs se dividen en dos niveles: los del **negocio** (que mide la aerolinea) y los del **modelo** (que decide si la solucion se despliega o se descarta).

### KPIs de negocio

Medidos sobre los 71 606 vuelos regulares de El Tepual, 2020-2025:

| KPI | Definicion | Valor actual | Meta propuesta |
|---|---|---|---|
| **Tasa de puntualidad** | % de vuelos con desvio ≤ 15 min sobre su horario habitual | **84.8%** | ≥ 88% |
| **Tasa de retraso severo** | % de vuelos con desvio > 60 min | **4.2%** | ≤ 3% |
| **Minutos de retraso acumulados** | Suma mensual de minutos de atraso — el proxy de costo | **≈ 12 500 min/mes** (≈ 208 h) | −15% |
| **Concentracion horaria del retraso** | Diferencia de puntualidad entre la primera y la ultima ola del dia | **7.9 pts** (10.8% → 18.7%) | ≤ 5 pts |

El tercero es el que traduce a dinero: **208 horas de atraso al mes** en un solo aeropuerto. Multiplicado por el costo hora de una aeronave con tripulacion, es la cifra que justifica el proyecto.

### KPIs del modelo — y el umbral de decision

Un modelo predictivo solo se despliega si supera al metodo que ya existe (que hoy es "asumir que todos los vuelos tienen el mismo riesgo"). Los umbrales se fijaron **antes** de ver los resultados en el conjunto de prueba:

| KPI del modelo | Umbral para desplegar | Resultado obtenido | ¿Pasa? |
|---|---|---|---|
| **Lift sobre la tasa base** | ≥ 1.50 | **1.12** | ❌ No |
| **ROC-AUC en datos futuros** | ≥ 0.65 | **0.534** | ❌ No |
| **Brier score (calibracion)** | ≤ 0.13 | **0.119** | ✅ Si |
| **Aporte del clima al AUC** | > 0 | **−0.013** (lo empeora) | ❌ No |

**Tres de los cuatro KPIs no se cumplen, y el proyecto lo declara abiertamente.** El unico que pasa es la calibracion — el modelo es honesto sobre su propia incertidumbre, pero esa incertidumbre es demasiado alta para decidir con ella.

> Definir el umbral **antes** de mirar el test es lo que impide la trampa mas comun en ML aplicado: mover la meta hasta que el modelo la alcance.

---

## Metodologia — CRISP-DM

El proyecto sigue **CRISP-DM** (Cross-Industry Standard Process for Data Mining), y su caracteristica de ciclo — no de linea recta — fue determinante: el proyecto **volvio dos veces a fases anteriores** al descubrir problemas que no eran visibles al principio.

| Fase | Que se hizo aqui | Donde verlo |
|---|---|---|
| **1. Comprension del negocio** | Definicion del retraso como proxy de costo operativo; KPIs con umbral de despliegue fijado por adelantado; ventana de confianza de 7-14 dias | [Problema y objetivos](#problema-de-negocio-y-objetivos) · [KPIs](#kpis--como-se-mide-el-exito) |
| **2. Comprension de los datos** | Perfilado de las 12 columnas de la bitacora, conteo de nulos, cobertura del clima. **Aqui se descubrio que no existe hora programada** | [Preparacion de los datos](#preparacion-de-los-datos) · pipeline `data_inventory` |
| **3. Preparacion de los datos** | Imputacion de PMD, reconstruccion del horario de referencia, viento cruzado, LEFT JOIN por `fecha_cruce` | [Como se construye el retraso](#como-se-construye-el-retraso-si-el-dato-no-lo-trae) · [El cruce con el clima](#el-cruce-con-el-clima) |
| **4. Modelado** | K-Means (k=4), RandomForest, y para retrasos: GradientBoosting regularizado vs Regresion Logistica, sobre tres conjuntos de features | [Clustering](#modelado-no-supervisado--clustering) · [Score de Riesgo](#el-score-de-riesgo-operativo) |
| **5. Evaluacion** | Split temporal de tres tramos, importancia por permutacion, calibracion. **Contraste contra los umbrales definidos en la fase 1** | [El Score de Riesgo Operativo](#el-score-de-riesgo-operativo) |
| **6. Despliegue** | Modelos serializados en `data/06_models/`. **Decision explicita de NO desplegar** el modelo de retrasos | [Conclusiones](#conclusiones-y-decisiones-de-negocio) |

**Los dos ciclos de retroalimentacion que hubo que hacer:**

1. **Fase 2 → Fase 1.** Al perfilar los datos aparecio que la bitacora no trae hora programada. El objetivo original ("restar programada menos real") era irrealizable y hubo que redefinir el target antes de seguir.
2. **Fase 5 → Fase 3.** La primera evaluacion dio AUC 0.84 en entrenamiento y 0.49 en prueba — sobreajuste severo. Eso obligo a volver a la preparacion, detectar que se estaba usando la hora **real** en vez de la **programada**, y rehacer las features.

> Ese segundo ciclo es el mas instructivo del proyecto: **un error de preparacion de datos se disfrazo de buen resultado de modelado.** Solo la evaluacion con split temporal lo dejo al descubierto.

---

## Fuentes de datos y herramientas

### Fuentes de datos

| Fuente | Origen | Volumen | Licencia | Por que se eligio |
|---|---|---|---|---|
| **Bitacora de vuelos** | [datos.gob.cl — JAC](https://datos.gob.cl/dataset/operaciones-aeronaves) (`.parquet`) | 11 074 197 filas × 12 col, 137 MB | Datos abiertos de Gobierno de Chile | Unica serie con el detalle operacion a operacion; incluye la marca de tiempo que permite derivar el retraso |
| **Operaciones por aeropuerto** | [datos.gob.cl — JAC](https://datos.gob.cl/dataset/operaciones-aeronaves) (`.csv`) | Agregado mensual por aeropuerto | Datos abiertos | Aporta `cnt_operaciones`, la medida de congestion mensual que la bitacora no trae |
| **Clima diario** | [Meteostat, estacion 85799](https://meteostat.net/es/place/cl/puerto-montt?s=85799) | 2 192 dias, 11 columnas | Meteostat / fuentes NOAA-DWD | Cobertura completa del periodo sin dias faltantes y con direccion de viento, que es lo que permite calcular el viento cruzado |

Las tres son **publicas y de acceso abierto**; ninguna requiere credenciales ni contiene datos de pasajeros.

### Herramientas y su justificacion

| Herramienta | Para que | Por que esta, y no otra |
|---|---|---|
| **Git + GitHub** | Control de versiones y trabajo en paralelo del equipo | Permite que cada integrante trabaje en una rama sin pisar el trabajo del resto; el historial deja trazable **quien** cambio **que** y **por que** — clave para una defensa individual |
| **Kedro** | Orquestacion del pipeline | Convierte cada transformacion en un nodo con entradas y salidas declaradas. Sin esto, el proyecto seria un notebook de 800 lineas imposible de revisar entre varias personas |
| **Catalogo de datos (`catalog.yml`)** | Registro central de datasets | Nadie escribe rutas de archivo a mano. Un integrante puede cambiar donde vive un dato sin romperle el codigo a los demas |
| **`parameters_*.yml`** | Configuracion separada del codigo | Cambiar de aeropuerto o de umbral de retraso no requiere tocar Python — importa para que el PoC sea auditable |
| **uv** | Entorno reproducible (`uv.lock`) | Garantiza que los cinco integrantes corran exactamente las mismas versiones; elimina el "en mi maquina funciona" |
| **pytest** | Pruebas automatizadas | 16 tests cubren la logica de reconstruccion horaria, que es la parte del codigo donde un error no se ve en las metricas agregadas |
| **Jupyter** | Informe ejecutable ([notebooks/](notebooks/)) | Permite que un evaluador recorra el proyecto paso a paso sin leer el codigo fuente |

---

## Arquitectura y Pipeline

El proyecto esta construido con **Kedro**, un framework que convierte cada transformacion en un nodo trazable y reproducible. El flujo completo tiene **18 nodos** que corren en ~146 segundos.

```
vuelos_raw (11M filas)                        clima_diario_scte.csv (2 192 dias)
    │                                                      │
    ├──────────── BASELINE ────────────┐                   │
    ▼                                  │                   │
preprocess_vuelos                      │                   │
    ▼                                  │                   │
join_con_aeropuertos                   │                   │
    │ (03_primary)                     │                   │
    ├──▶ generate_eda_plots            │                   │
    ├──▶ perfil_aeropuertos ── cluster_aeropuertos ─▶ kmeans.pkl
    │                       └─ evaluar_clustering  ─▶ curva del codo
    ├──▶ eda_avanzado                  │                   │
    ├──▶ train_clasificador  ─▶ random_forest.pkl          │
    └──▶ evaluar_clasificador ─▶ CV-5 + curva ROC          │
                                                           │
    └──────────── RETRASOS_ML ─────────┐                   │
    ▼                                  │                   ▼
filtrar_vuelos_scte              preparar_clima_scte  ◀────┘
 (11M → 80 323 filas)             (limpieza + viento cruzado)
    ▼                                  │
construir_target_retraso               │
 (reconstruye el horario)              │
    └────────▶ cruzar_vuelos_clima ◀───┘
                    │ (03_primary, LEFT JOIN por fecha_cruce)
                    ├──▶ train_modelo_retrasos  ─▶ modelo_retrasos.pkl
                    └──▶ evaluar_impacto_clima  ─▶ 4 graficos + tabla de efecto
```

**Pipelines registrados:**

| Pipeline | Que hace |
|---|---|
| `data_inventory` | Inventario de archivos en `data/01_raw/` |
| `data_processing` | Limpieza, JOIN y EDA basico |
| `ml` | EDA avanzado, clustering y clasificacion (baseline) |
| `retrasos_ml` | PoC de riesgo de retraso en Puerto Montt con datos de clima |

El baseline **no fue modificado**. `retrasos_ml` es un pipeline aparte que parte de la misma fuente cruda.

---

## Como ejecutar el proyecto

```bash
# Clonar e instalar
git clone https://github.com/donMixho/OperacionesAeronaves.git
cd OperacionesAeronaves
uv sync

# Descargar las fuentes crudas a data/01_raw/
#   - bitacora-vuelos.parquet y operaciones-aeropuertos.csv desde datos.gob.cl
#   - clima diario de Puerto Montt (se baja solo):
python scripts/descargar_clima_scte.py

# Ejecutar todo (~146 segundos)
uv run kedro run

# O por partes
uv run kedro run --pipeline=data_processing
uv run kedro run --pipeline=ml
uv run kedro run --pipeline=retrasos_ml   # ~21 segundos

# Pruebas (16 tests sobre la reconstruccion del horario)
uv run pytest tests/pipelines/retrasos_ml/
```

**Informe ejecutable.** El notebook [`notebooks/informe_ml_operaciones.ipynb`](notebooks/informe_ml_operaciones.ipynb) recorre el proyecto completo siguiendo las seis fases de CRISP-DM, llamando a **las mismas funciones** que ejecuta el pipeline — no duplica logica. Corre en ~1 minuto:

```bash
uv run jupyter lab notebooks/informe_ml_operaciones.ipynb
```

> **Semilla fija:** todos los modelos usan `random_state=42` para garantizar resultados identicos entre ejecuciones.

---

## Preparacion de los datos

Las fuentes crudas tienen problemas tipicos de datos reales. Asi los resolvimos:

**`bitacora-vuelos.parquet`** — 11 074 197 filas, 12 columnas

| Problema | Solucion |
|---|---|
| `numero_vuelo`: 457 687 nulos (4.1%) | Se rellena con la cadena `"DESCONOCIDO"` |
| `pmd`: 275 115 nulos (2.5%) | Se crea un flag `pmd_fue_imputado` y se imputa con la **mediana agrupada por modelo de avion** (si el modelo entero no tiene datos, se usa la mediana global) |
| `mes_id` no existia | Se extrae de `dt_operacion` en formato YYYYMM |
| `internacional_domestico` no existia | Se mapea desde el booleano `es_internacional` → `'I'` o `'D'` |

**`operaciones-aeropuertos.csv`** — estadisticas mensuales agregadas por aeropuerto

Se une a la bitacora con un LEFT JOIN sobre `[aeropuerto_oaci, mes_id, internacional_domestico]`.
Resultado: **99.996% de cobertura** — solo 400 filas de 11 millones quedaron sin match.

**`clima_diario_scte.csv`** — clima diario de Puerto Montt, estacion Meteostat 85799

2 192 dias (2020-01-01 a 2025-12-31), sin un solo dia faltante. Tres columnas (`snow`, `wpgt`, `tsun`) vienen 100% vacias para esta estacion y se descartan automaticamente. `prcp` tiene 82% de cobertura y los nulos se tratan como "no llovio", que es la convencion de Meteostat.

---

## Analisis Exploratorio — Graficos clave

### ¿Como se distribuye el trafico por aeropuerto?

![Top 15 aeropuertos](images/eda_01_top_aeropuertos.png)

**¿Para que sirve?** Ver rapidamente cuales aeropuertos concentran el poder del sistema aereo chileno.

**¿Que descubrimos?** Santiago (SCEL) opera mas del doble que el segundo aeropuerto (Tobalaba, SCTB). El sistema es extremadamente asimetrico: el aeropuerto numero uno tiene mas trafico que los siguientes 14 juntos. Esto tiene implicancias directas en politica de infraestructura.

---

### ¿La imputacion del PMD altera la realidad del dato?

![PMD imputado vs original](images/eda_03_pmd_imputado.png)

**¿Para que sirve?** Validar que rellenar los valores faltantes de PMD (Peso Maximo de Despegue) con la mediana del modelo de avion no introduce sesgo en el analisis.

**¿Que descubrimos?** Las dos curvas — valores originales e imputados — son casi identicas. El pequeño "pico" que aparece en los valores imputados es normal y esperable: **cuando se imputa con una mediana, todos los registros de un mismo modelo de avion reciben exactamente el mismo valor** (el valor central de ese grupo). Eso apila muchas observaciones en un solo punto del eje y crea una barra alta, pero **no distorsiona la distribucion real** — la masa sigue estando donde estaba. La fisica del avion no cambia: simplemente usamos la mejor estimacion disponible para ese modelo en vez de descartar el registro.

---

### ¿Que variables se relacionan entre si?

![Correlacion variables numericas](images/eda_07_correlacion_numericas.png)

**¿Para que sirve?** Entender si el peso del avion, el mes y el volumen de operaciones tienen alguna relacion entre si antes de entrenar el modelo.

**¿Que descubrimos?** El PMD tiene una correlacion positiva moderada con `es_internacional` (r = 0.31): los aviones mas pesados tienden a usarse en rutas internacionales. El flag de imputacion tiene correlacion negativa con PMD (-0.24), lo que confirma que los datos faltantes ocurren principalmente en aeronaves livianas (que tienen menor registro formal). El resto de variables son practicamente independientes entre si — buena señal para el modelo.

---

## Modelado No Supervisado — Clustering

**Pregunta:** ¿Existen grupos naturales de aeropuertos segun su comportamiento operativo?

Agrupamos los 69 aeropuertos usando **K-Means** sobre cinco variables: volumen de vuelos, porcentaje internacional, PMD mediano, numero de aerolineas y promedio de operaciones mensuales (todo transformado con logaritmo para manejar la escala).

Probamos k=2 hasta k=8. El optimo matematico es **k=4** — mayor Silhouette Score (0.420) y punto de inflexion en la curva del codo.

![Clusters aeropuertos](images/ml_01_clusters_aeropuertos.png)

**¿Que grupos encontramos?**

| Cluster | Aeropuertos | Caracter |
|---|---|---|
| 0 — Regionales | 32 | Poco trafico, aviones livianos, casi sin vuelos internacionales |
| 1 — Pesados mixtos | 16 | Volumen medio-alto, aviones de carga o turbopropulsores |
| 2 — Hub global | 1 (SCEL) | El unico aeropuerto con perfil verdaderamente internacional (43% de sus vuelos) |
| 3 — Hubs domesticos | 20 | Alto volumen, aviones livianos, trafico casi 100% nacional |

> SCEL forma un cluster propio. Es tan distinto al resto que el algoritmo lo aisla solo — lo cual tiene todo el sentido operativo.

---

## Modelado Supervisado — Clasificacion

**Pregunta:** ¿Podemos predecir si un vuelo es internacional solo con sus datos operativos?

**Por que esta variable objetivo:** `es_internacional` es la distincion mas relevante para planificar infraestructura aeroportuaria — define requisitos de aduana, rampa, gate y personal. Si podemos predecirla con precision, podemos anticipar necesidades antes de que ocurran.

**El desafio:** solo el 12.8% de los vuelos son internacionales. Para que el modelo no ignore esta clase minoritaria, entrenamos con `class_weight="balanced"`.

**Modelo:** RandomForestClassifier — 200 arboles, profundidad maxima 12, semilla 42.
**Muestra:** 500 000 filas (estratificadas), split 80/20.

### ¿Que variables importan mas?

![Importancia de variables](images/ml_02_importancia_features.png)

**¿Para que sirve?** Ver cuales datos son los que realmente le permiten al modelo distinguir un vuelo internacional.

**¿Que descubrimos?** El **aeropuerto** es el predictor dominante — tiene sentido: solo ciertos aeropuertos tienen rutas internacionales. El **PMD** (peso del avion) es el segundo predictor: los aviones mas pesados casi siempre vuelan mas lejos. El mes y el año tambien aportan — el trafico internacional tiene estacionalidad y ha crecido con los años.

### ¿Que tan bien discrimina el modelo?

![Curva ROC](images/ml_05_roc_curve.png)

**¿Para que sirve?** Medir la capacidad del modelo para separar vuelos internacionales de domesticos en cualquier umbral de decision.

**¿Que descubrimos?** Un **ROC-AUC de 0.964** significa que el modelo clasifica correctamente el 96.4% de los pares vuelo-internacional vs. vuelo-domestico. La validacion cruzada de 5 particiones confirma que este resultado es estable (0.964 ± 0.001) — no es suerte de una sola particion.

### Metricas del modelo

| Metrica | Valor | Que significa en la practica |
|---|---|---|
| **ROC-AUC** | 0.964 | Excelente capacidad de discriminacion global |
| **Recall** | 0.965 | Detecta el 96.5% de los vuelos internacionales reales |
| **Precision** | 0.465 | De cada 10 predichos como "internacional", ~5 realmente lo son |
| **F1** | 0.628 | Balance entre precision y recall |
| **Accuracy** | 0.853 | 8 de cada 10 predicciones son correctas |

> El Recall alto es una decision deliberada: en planificacion de infraestructura, es peor no detectar un vuelo internacional (y quedarse sin gate) que sobreestimarlo.

---

# Prediccion de Retrasos — PoC Puerto Montt

## Impacto de Negocio — por que predecir retrasos

El modelo base predice *que tipo* de vuelo es. Util, pero no mueve la aguja economica. **El retraso si.** Cada minuto que un avion se demora consume combustible en tierra, horas de tripulacion, uso de gate y compensaciones a pasajeros. Por eso el retraso se usa aqui como **variable proxy del costo operativo**: no tenemos las facturas de la aerolinea, pero si tenemos los minutos, y los minutos cuestan dinero.

**Por que solo Puerto Montt.** Procesar los 69 aeropuertos multiplicaria el costo computacional sin agregar conocimiento nuevo en esta etapa. El Tepual (SCTE) es un buen banco de pruebas: tiene volumen suficiente (80 323 operaciones regulares en el periodo), 22 aerolineas, 263 numeros de vuelo distintos, y un clima lo bastante hostil como para que la hipotesis del clima sea razonable — **llueve el 44.9% de los dias**. Escalar a todo Chile es un problema de infraestructura, no de metodo (ver [Trabajo futuro](#trabajo-futuro)).

**Por que solo 2020–2026.** Se descartaron los datos de 1999 a 2019. Eso baja el parquet de 11 millones de filas a 80 mil, acelera el entrenamiento, y sobre todo evita que el modelo aprenda patrones de una infraestructura que ya no existe.

---

## Como se construye el retraso si el dato no lo trae

Esta es la parte mas importante de esta seccion y la primera pregunta que corresponde hacer en una defensa.

**El plan original era restar la hora programada de la hora real.** Al abrir el parquet quedo claro que eso no se puede hacer: la bitacora JAC tiene **una sola marca de tiempo**.

```
dt_operacion: timestamp[us, tz=America/Santiago]   ← la hora REAL de la operacion
aeropuerto_oaci, tipo_operacion, aeropuerto_dgac_orig_dest,
aerolinea_dgac, numero_vuelo, actividad_cod, matricula,
modelo_avion, modelo_avion_desc, es_internacional, pmd
```

No hay hora programada. No hay itinerario publicado. Restar "programada − real" es imposible con esta fuente.

### La solucion: reconstruir el horario desde la propia bitacora

Un vuelo regular opera a la misma hora todos los dias. Si LATAM 61 aterriza a las 11:11, 11:32, 11:33, 11:05, 11:18… su **horario habitual** es ~11:15. El vuelo que ese dia llego a las 12:40 se retraso. La referencia no es el itinerario oficial, pero es una reconstruccion razonable de el.

El horario de referencia se calcula como la **mediana del slot habitual** de cada combinacion de `aerolinea + numero_vuelo + tipo_operacion + mes + dia de la semana`. Tres detalles hicieron la diferencia entre una medida util y ruido puro:

**1. El reloj es circular.** Un vuelo de las 23:50 y otro de las 00:10 distan 20 minutos, no 1420. Sin corregir esto, los vuelos nocturnos destruian la medida: un vuelo de JetSmart programado a las 22:00 que a veces despega 00:31 producia una "mediana" a mitad de la tarde, una hora en la que ese vuelo nunca opero.

**2. Un mismo numero de vuelo puede tener dos slots.** LATAM 61 llega **~07:16 los sabados** y **~19:34 lunes y miercoles**. Promediar ambos da un horario que no existe. El algoritmo separa los slots buscando huecos de mas de 120 minutos en el reloj circular, y compara cada operacion contra el centro del slot mas cercano.

**3. Las aerolineas programan por dia de la semana.** Agregar el dia de la semana a la agrupacion fue lo que mas mejoro la medida.

El efecto acumulado de las tres correcciones, medido sobre las mismas 80 323 operaciones:

| Version del calculo | Desv. estandar | p05 | p95 | % "retrasados" |
|---|---|---|---|---|
| Mediana simple de la hora | 161.8 min | −248 min | +270 min | 31.3% |
| + reloj circular | 155.4 min | −241 min | +253 min | 30.4% |
| + deteccion de slots | 71.3 min | −90 min | +110 min | 26.4% |
| + dia de la semana **(version final)** | **57.5 min** | **−27.5 min** | **+50.5 min** | **14.4%** |

**Como sabemos que la reconstruccion es correcta.** El criterio de validacion es la **cola izquierda**: un avion comercial casi nunca despega mucho *antes* de su horario — los pasajeros no han embarcado. Si la referencia estuviera mal, veriamos multitud de vuelos "adelantados" horas, que es exactamente lo que pasaba en las primeras versiones (p05 = −248 min, es decir mas de 4 horas antes: imposible).

Notese tambien como la tasa de "retraso" cae de 31.3% a 14.4% al ir corrigiendo el calculo: **mas de la mitad de los retrasos que mostraba la version ingenua eran errores de medicion, no retrasos.**

Sobre los 71 606 vuelos que finalmente conservan horario de referencia, el desvio se distribuye asi:

| p05 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|
| −29 min | −5.5 min | **0 min** | +7 min | +52 min |

La distribucion esta centrada en cero, la cola izquierda es corta y plausible, y la cola derecha es larga. **Esa asimetria es la firma de un retraso real**: los vuelos se atrasan mucho y se adelantan poco. Si la medida fuera ruido, seria simetrica.

### El target

- Se conservan solo los vuelos de **transporte publico regular** (`actividad_cod = 'U'`): sin itinerario no existe la nocion de retraso.
- Se exigen al menos 3 observaciones por grupo para aceptar su horario de referencia. Esto deja **71 606 vuelos (89.1%)** del recorte.
- `retraso = 1` si el vuelo opero **mas de 15 minutos** despues de su horario habitual.
- **Tasa de retraso global: 15.2%.**

### Lo que esta medida NO es

Honestidad sobre los limites, porque van a preguntar:

- **Mide irregularidad, no retraso oficial.** Si un vuelo sale sistematicamente 20 minutos tarde todos los dias, su horario habitual *incluye* esos 20 minutos y el modelo lo considera puntual. La medida captura desviaciones del comportamiento tipico, no del itinerario publicado.
- **Es conservadora.** Si un dia entero se retrasa completo, parte de ese retraso se absorbe en la mediana del grupo. Subestima, no sobreestima.
- **Depende de la regularidad.** Funciona para LATAM, Sky y JetSmart; no funcionaria para aviacion general o vuelos charter, que por eso se excluyen.
- **La hora programada que usa el modelo se calcula con datos del mismo mes.** En produccion se usaria el itinerario publicado; aqui se reconstruye con informacion que en rigor todavia no estaria disponible. Es un sesgo optimista leve, y aun asi el modelo apenas supera el azar — con el itinerario real el resultado no seria mejor por este motivo.

> **Lo que si esta correctamente aislado:** el modelo nunca ve la hora *real* de la operacion, que es justo lo que debe predecir. Solo recibe la hora *programada*. Confundir ambas inflaria las metricas de forma espectacular y falsa.

---

## El cruce con el clima

El clima diario de Meteostat (estacion 85799, Puerto Montt) se une a los vuelos con un **LEFT JOIN por un solo campo**: `fecha_cruce`, en formato `YYYY-MM-DD`. Ambos lados se normalizan eliminando hora y zona horaria, justamente para que el merge sea barato y no colapse la memoria.

**Resultado del cruce: 71 606 vuelos, 0 filas sin clima.** Cobertura perfecta, porque el CSV cubre los 2 192 dias del periodo sin faltantes.

Ademas de las variables crudas (temperatura, lluvia, viento, presion) se derivan dos que si tienen sentido aeronautico:

- **Viento cruzado.** Un viento de 40 km/h alineado con la pista casi no molesta; el mismo viento de costado puede cerrar el aeropuerto. El Tepual tiene pista **17/35** (rumbo 170°), asi que el viento se descompone en su componente perpendicular: `viento_cruzado = wspd × |sin(wdir − 170°)|`.
- **Amplitud termica** (`tmax − tmin`), como proxy de inestabilidad atmosferica.

> **Nota metodologica:** tambien se probo el clima **horario** de Meteostat (temperatura, punto de rocio, humedad y codigo de condicion hora a hora, cruzado con `merge_asof` contra la hora exacta de cada vuelo). No cambio la conclusion: la señal siguio siendo practicamente nula. Se mantuvo el cruce diario por ser mas simple, mas reproducible y tener mejor cobertura.

---

## ¿Cuanto retrasa realmente el clima?

Esta es la pregunta que motivo todo el pipeline. La respuesta tiene dos partes y ninguna es la esperada.

![Clima vs retraso](images/rt_01_clima_vs_retraso.png)

**¿Para que sirve?** Comparar directamente la tasa de retraso en dias con y sin mal tiempo. La linea naranja es la tasa base (15.2%): si las barras no se despegan de ella, el clima no esta haciendo nada.

**¿Que descubrimos?**

**1. La lluvia no retrasa vuelos en Puerto Montt.** Ninguna barra se separa de la tasa base. Un dia con mas de 15 mm de lluvia tiene una tasa de retraso de 15.19% — contra 15.14% en dias completamente secos. La diferencia es de **0.05 puntos porcentuales**, es decir, nada.

| Lluvia del dia | Vuelos | Tasa de retraso | vs. base |
|---|---|---|---|
| Sin lluvia | 40 602 | 15.14% | −0.05 pts |
| 0–1 mm | 9 377 | 14.38% | −0.82 pts |
| 1–5 mm | 10 165 | 15.86% | +0.66 pts |
| 5–15 mm | 8 117 | 15.57% | +0.38 pts |
| **> 15 mm** | 3 345 | **15.19%** | **−0.01 pts** |

La explicacion es operativa, no estadistica: **en Puerto Montt llueve el 44.9% de los dias**. Un aeropuerto que opera bajo lluvia casi la mitad del año esta diseñado, equipado y entrenado para la lluvia. La lluvia ahi no es una anomalia — es la condicion normal.

**2. El viento cruzado si retrasa, pero solo en el extremo.** Aqui si aparece una señal, y es fuerte:

| Viento cruzado | Vuelos | Tasa de retraso | vs. base |
|---|---|---|---|
| < 5 km/h | 32 583 | 14.65% | −0.54 pts |
| 5–10 km/h | 28 831 | 15.26% | +0.07 pts |
| 10–20 km/h | 9 064 | 16.55% | +1.35 pts |
| 20–30 km/h | 947 | 16.68% | +1.49 pts |
| **> 30 km/h** | **181** | **25.97%** | **+10.77 pts** |

Con mas de 30 km/h de viento perpendicular a la pista, la tasa de retraso **casi se duplica**: de 14.7% a 26.0%. El efecto es real y tiene sentido fisico — es el viento cruzado, no la lluvia, lo que limita un aterrizaje.

**Pero ese escenario ocurre en 181 de 71 606 vuelos: el 0.25%.** Y ahi esta la clave de todo el proyecto.

---

## El Score de Riesgo Operativo

El modelo entrega `predict_proba` — un numero continuo de 0% a 100% — en vez de una etiqueta si/no. Asi el gerente elige el umbral segun lo que le cueste cada tipo de error, en lugar de heredar el 50% por defecto.

### Como se evalua (y por que no con un split aleatorio)

El modelo se ofrece como pronostico a futuro, asi que evaluarlo con un split aleatorio seria hacer trampa: pondria vuelos del mismo dia en entrenamiento y en prueba, y el modelo veria el clima de un dia que ya conoce. El split es **temporal y de tres tramos**:

| Tramo | Periodo | Vuelos | Para que |
|---|---|---|---|
| Entrenamiento | 2020–2023 | 42 067 | Aprender |
| Validacion | 2024 | 14 850 | Elegir modelo e hiperparametros |
| **Prueba** | **2025+** | **14 689** | Se toca **una sola vez** |

### El experimento que responde la pregunta

Cada modelo se entrena **tres veces**: solo con variables de operacion, solo con clima, y con ambas. Si sumar el clima no mueve el AUC, el clima no explica el retraso. Resultados sobre 2025 (tasa base 13.8%):

| Variables | Modelo | AUC train | AUC validacion | **AUC test** |
|---|---|---|---|---|
| **Solo operacion** | **GradientBoosting** | 0.579 | **0.545** | **0.534** ✅ |
| Solo operacion | Regresion Logistica | 0.555 | 0.540 | 0.533 |
| Solo clima | GradientBoosting | 0.565 | 0.510 | **0.469** |
| Solo clima | Regresion Logistica | 0.526 | 0.519 | 0.488 |
| Clima + operacion | GradientBoosting | 0.587 | 0.544 | 0.521 |
| Clima + operacion | Regresion Logistica | 0.559 | 0.536 | 0.522 |

**Tres lecturas, todas incomodas y todas honestas:**

1. **Con solo clima, el modelo queda por debajo del azar** (AUC 0.469 y 0.488, contra 0.5 de una moneda). El clima diario, por si solo, no contiene informacion util sobre si un vuelo se va a retrasar.
2. **Agregar el clima a las variables de operacion empeora el modelo** (0.534 → 0.521). No es un error de codigo: es el comportamiento clasico de una variable que aporta ruido en vez de señal, y que el modelo termina usando para memorizar el pasado.
3. **El mejor modelo llega a AUC 0.534.** Eso es apenas mejor que el azar. El pipeline selecciona automaticamente el ganador por el año de validacion, y **el ganador no usa clima**.

![Importancia de variables](images/rt_02_importancia.png)

**¿Para que sirve?** Medir cuanto pierde el modelo si se destruye una variable (importancia por permutacion). A diferencia de la importancia Gini, esta no premia a las variables con muchos valores distintos, asi que comparar clima contra operacion es justo. Las barras negativas indican variables que **empeoran** el modelo.

**¿Que descubrimos?** Las dos unicas variables con aporte real son **la hora programada** y **la aerolinea** — ambas operativas (azul). Todas las variables de clima (naranjo) estan pegadas a cero o en negativo. `amplitud_termica` es la peor de todas: el modelo la uso para memorizar el pasado y en 2025 le costo AUC.

![Retraso por hora del dia](images/rt_03_hora_del_dia.png)

**¿Para que sirve?** Ver el unico patron que el modelo si logra capturar: como se acumula el retraso a lo largo de la jornada.

**¿Que descubrimos?** Un vuelo programado a las 06:00 tiene 10.8% de probabilidad de retrasarse; uno programado a las 22:00, 18.7%. El avion de la tarde hereda los atrasos de la mañana — es propagacion en la rotacion de la flota, no clima.

> **Un sesgo que corregimos en el camino:** una version preliminar de este analisis media la hora *real* de la operacion en vez de la programada, y mostraba un efecto mucho mas dramatico (7.6% a las 06:00 contra 21.7% a las 21:00). Ese efecto era en parte artificial: **un vuelo retrasado se mueve solo hacia horas mas tardias**, asi que las horas de la noche concentran retrasos por construccion. Usando la hora programada — la unica que un planificador conoce por anticipado — el efecto real es la mitad.

![ROC y calibracion](images/rt_04_roc_calibracion.png)

**¿Para que sirve?** El panel izquierdo mide la discriminacion; el derecho verifica que el score se pueda leer como porcentaje (que un score de 30% corresponda de verdad a un 30% de vuelos retrasados).

**¿Que descubrimos?** La curva ROC se despega apenas de la diagonal del azar (AUC 0.534). El panel de calibracion es aun mas elocuente: **todos los scores del modelo caben entre 13.5% y 19%**. El modelo esta prediciendo casi la tasa base para todos los vuelos, porque no encontro nada que le permita distinguirlos. Esta bien calibrado — y es casi inutil para decidir.

### Metricas finales del Score de Riesgo

| Metrica | Valor | Que significa |
|---|---|---|
| ROC-AUC (test 2025) | **0.534** | Apenas sobre el azar (0.5) |
| Average Precision | 0.155 | Contra una tasa base de 0.138 |
| Lift vs azar | **1.12** | Solo un 12% mejor que adivinar |
| Brier score | 0.119 | Bien calibrado, pero sobre un rango estrecho |
| Tasa base en test | 13.8% | El 13.8% de los vuelos de 2025 se retrasaron |

**Veredicto honesto: este modelo no esta listo para produccion.** Un lift de 1.12 no sostiene una decision operativa. Se entrega documentado como lo que es — evidencia de que la hipotesis inicial no se cumple con estos datos, y una arquitectura lista para recibir mejores insumos.

---

## Ventana de Confianza — 7 a 14 dias

Aunque el modelo actual no discrimina lo suficiente, la regla de negocio queda definida y vigente para cuando si lo haga. **El Score de Riesgo solo es confiable en una ventana tactica de 7 a 14 dias hacia el futuro.** Mas alla de ese horizonte el margen de error supone riesgos inaceptables para una decision gerencial.

Las razones son tres:

1. **Concept drift.** Los patrones operativos cambian: itinerarios nuevos, flotas distintas, huelgas, cambios regulatorios. El modelo entrenado con 2020–2023 ya perdio precision en 2025 (AUC de validacion 0.545 contra 0.534 en prueba). Un modelo entrenado sobre el pasado envejece.
2. **El pronostico meteorologico se degrada.** Mas alla de ~10 dias, un pronostico de clima no es mejor que la climatologia historica. Alimentar el modelo con un pronostico a 30 dias es alimentarlo con ruido.
3. **El itinerario todavia no es firme.** Mas alla de dos semanas, las aerolineas aun ajustan horarios, y el horario programado es justamente la variable mas predictiva del modelo.

> Este disclaimer **no es una limitacion tecnica que ocultar** — es parte del diseño del producto. Un score que se presente como valido a 6 meses seria irresponsable.

---

## Etica, sesgos y privacidad

Un modelo que decide sobre operaciones aereas toca a aerolineas, trabajadores y pasajeros. Esta seccion documenta que puede salir mal.

### 1. Privacidad — el riesgo real esta en la matricula

La bitacora **no contiene datos de pasajeros**: no hay nombres, documentos, edades ni destinos individuales. Es un registro de operaciones de aeronaves. En ese sentido el riesgo de privacidad es bajo.

**Pero hay una excepcion importante, y es la columna `matricula`.**

La matricula identifica de forma unica a una aeronave, y el registro de aeronaves de la DGAC es publico: **de una matricula se puede llegar a su propietario**. El dataset completo contiene **13 775 matriculas distintas**, y **el 44.9% de las operaciones de todo Chile (4 967 070 vuelos) son aviacion no regular** — privada, general o de instruccion.

Eso significa que, cruzando `matricula + fecha + aeropuerto`, **cualquiera puede reconstruir el itinerario historico de la aeronave de una persona identificable**: cuando viajo, desde donde y con que frecuencia. Es un dato de movimiento personal disfrazado de dato operativo. La JAC lo publica agregado en un dataset tecnico, pero la re-identificacion es trivial para quien tenga acceso al registro de matriculas.

**Que hicimos al respecto:**

- El pipeline filtra `actividad_cod == 'U'` (transporte publico regular), lo que **excluye toda la aviacion privada y de instruccion**. El filtro se introdujo por una razon tecnica — sin itinerario no hay retraso que medir — pero tiene el efecto colateral de eliminar el grupo con riesgo de re-identificacion.
- `matricula` **no se usa como feature** en ningun modelo. No entra en la matriz de entrenamiento.
- Quedan 478 matriculas en el recorte, todas de flotas comerciales (LATAM, Sky, JetSmart), donde la matricula identifica a una empresa, no a una persona.

**Recomendacion para quien extienda este proyecto:** si se levanta el filtro de `actividad_cod` para analizar aviacion general, la matricula debe hashearse o agregarse antes de cualquier publicacion. No es una formalidad — es la diferencia entre un dataset operativo y un registro de movimientos de personas.

### 2. Sesgos identificados en los datos

**a) Sesgo de seleccion contra operadores pequeños.** Para estimar un horario de referencia se exigen al menos 3 operaciones del mismo vuelo en el mes. Eso excluye **8 717 vuelos (10.9%)** — y no los excluye al azar:

| Aerolinea | Vuelos | % incluido en el modelo |
|---|---|---|
| LXP (LATAM Express) | 21 585 | 99.9% |
| SKU (Sky) | 26 776 | 98.4% |
| LAN (LATAM) | 18 491 | 98.9% |
| JAT (JetSmart) | 9 105 | 98.6% |
| 1D (operador menor) | 337 | **79.2%** |
| 6R (operador menor) | 448 | **85.7%** |
| 60F (operador menor) | 277 | **87.4%** |

Las grandes entran casi completas; los operadores pequeños pierden hasta **1 de cada 5 vuelos**. **El modelo representa peor a quien menos vuela.** Si este score se usara para asignar recursos o evaluar desempeño, penalizaria sistematicamente a los operadores regionales — justo los que menos capacidad tienen para defenderse de una medicion injusta.

**b) Sesgo de regimen COVID.** El periodo arranca en 2020, el año del cierre aereo. El volumen de 2020 (6 168 vuelos) es **menos de la mitad** que el de 2024 (14 850). Un aeropuerto operando al 40% de capacidad tiene una dinamica de retrasos completamente distinta a uno saturado. El modelo entrena con ambos regimenes como si fueran el mismo fenomeno.

**c) Sesgo del target reconstruido — el mas incomodo.** Como el horario de referencia se calcula desde la operacion real, **una aerolinea que sale sistematicamente 20 minutos tarde todos los dias aparece como puntual**: esos 20 minutos quedan incorporados en su horario "habitual". El target premia la impuntualidad **consistente** y castiga la variabilidad. Una aerolinea puntual pero irregular puede salir peor evaluada que una cronicamente atrasada pero predecible. **Usar esta medida para comparar aerolineas entre si seria un error metodologico grave.** Sirve para detectar dias anomalos, no para rankear operadores.

**d) Sesgo de imputacion.** El 2.5% de los PMD faltantes se imputa con la mediana del modelo de avion. Las aeronaves de modelos raros, sin datos en su grupo, reciben la mediana global — quedan artificialmente "promedio". El flag `pmd_fue_imputado` permite rastrear y auditar cuales son.

**e) Sesgo geografico.** El PoC cubre **un** aeropuerto. El Tepual tiene un clima, una pista y una mezcla de trafico particulares. La conclusion "la lluvia no retrasa vuelos" **es valida para Puerto Montt y no debe extrapolarse** a Punta Arenas, Iquique ni Santiago sin repetir el analisis.

### 3. Riesgos eticos de uso

| Riesgo | Por que importa | Mitigacion adoptada |
|---|---|---|
| **Uso punitivo contra trabajadores** | Un "Score de Riesgo" por vuelo podria usarse para evaluar tripulaciones o penalizar turnos. Con AUC 0.534, cualquier decision laboral basada en el seria practicamente azar con apariencia de objetividad | Se documenta explicitamente que el modelo **no discrimina** lo suficiente; el README desaconseja su despliegue |
| **Sesgo de automatizacion** | Un numero de 0 a 100% proyecta una precision que el modelo no tiene. La gente confia mas en un porcentaje que en una opinion | El grafico de calibracion muestra que **todos los scores caben entre 13.5% y 19%** — la falta de poder discriminante queda a la vista, no escondida en una metrica |
| **Discriminacion contra operadores pequeños** | Ver sesgo (a) | Se cuantifica la brecha de inclusion por aerolinea; se recomienda no usar el score para asignar recursos entre operadores |
| **Uso dual / vigilancia** | Los datos de movimiento de aeronaves sirven para seguir personas | Filtro a aviacion comercial regular; matricula excluida de las features |
| **Presentar un modelo fallido como exitoso** | Es el riesgo etico **mas probable** en un proyecto academico con nota de por medio | Los umbrales de despliegue se fijaron antes de ver el test, y se reporta que 3 de 4 no se cumplen |

### 4. La decision etica central del proyecto

Habia dos caminos al llegar a los resultados finales. El primero: buscar un split que diera mejores metricas, elegir el mejor de varios experimentos y presentar un AUC favorable. Con un split aleatorio en vez de temporal, las cifras de este proyecto habrian sido notablemente mejores — y falsas, porque el modelo habria visto vuelos del mismo dia en entrenamiento y en prueba.

El segundo: **evaluar con split temporal, reportar que el clima no funciona y explicar por que.**

Se tomo el segundo. Un modelo con lift 1.12 presentado como listo para produccion no es un error tecnico: es una afirmacion falsa sobre la que alguien podria tomar decisiones operativas reales. **El valor de este proyecto esta en haber medido bien algo que resulto no funcionar, no en haber forzado un resultado que si.**

---

## Conclusiones y Decisiones de Negocio

**1. El sistema aereo chileno es un monocentro.**
SCEL concentra el trafico internacional de forma casi monopolica. Cualquier decision de politica de expansion de rutas internacionales pasa obligatoriamente por Santiago — no hay alternativa real en el corto plazo.

**2. Los aeropuertos del Cluster 3 son los grandes olvidados del analisis tradicional.**
Aeropuertos como Tobalaba (SCTB) o Concepcion (SCIE) tienen volumenes altisimos de operaciones, pero casi toda es aviacion general o entrenamiento. No son "grandes" por tener muchos pasajeros — son grandes por intensidad de uso. Requieren regulacion diferenciada.

**3. La imputacion de PMD es confiable.**
El 2.5% de registros con PMD faltante fue imputado con la mediana del modelo de avion. La distribucion resultante es indistinguible de la original.

**4. El clasificador de vuelos internacionales puede operar en produccion.**
Con ROC-AUC de 0.964 estable en CV-5, esta listo para aplicarse a registros nuevos. El modelo serializado esta en `data/06_models/random_forest.pkl`.

**5. En Puerto Montt, la lluvia no retrasa vuelos — el viento cruzado si, pero casi nunca ocurre.**
Es el hallazgo central del PoC y va contra la intuicion. Un aeropuerto que opera bajo lluvia el 44.9% de los dias esta adaptado a la lluvia. El viento cruzado sobre 30 km/h casi duplica la tasa de retraso (14.7% → 26.0%), pero se da en el 0.25% de los vuelos: **demasiado raro para que un modelo lo aproveche, suficientemente severo para justificar un protocolo operativo especifico.**

**6. La recomendacion de negocio no es un modelo — es un umbral.**
Dado que el efecto del clima se concentra en un extremo raro, la accion rentable no es desplegar un modelo predictivo, sino **una alerta simple: cuando el pronostico indique viento cruzado sobre 30 km/h en El Tepual, activar el protocolo de contingencia.** Eso captura casi todo el valor del clima sin la infraestructura de un modelo.

**7. El retraso en Puerto Montt es un problema de red, no de clima.**
Las unicas variables predictivas son la hora programada y la aerolinea. El retraso se acumula durante el dia (10.8% a las 06:00 → 18.7% a las 22:00): llega desde Santiago en el avion, no desde el cielo de Puerto Montt.

---

## Trabajo futuro

Ordenado por impacto esperado, no por facilidad:

**1. Conseguir el itinerario real.** Es la mejora con mas potencial y la que resolveria la limitacion de fondo. Con la hora programada oficial (de una API de itinerarios o de los propios sistemas de la aerolinea), el target dejaria de ser una reconstruccion y el retraso seria exacto.

**2. Modelar la propagacion de red.** Los datos muestran que el retraso llega en el avion desde el aeropuerto anterior. Encadenar las operaciones por `matricula` permitiria construir la variable mas prometedora: *¿venia atrasado el avion que opera este vuelo?*

**3. Incorporar visibilidad y techo de nubes.** El dataset diario de Meteostat no los trae, y en El Tepual son justamente los que cierran el aeropuerto. Los METAR de la DGAC si los tienen.

**4. Escalar a todo Chile.** El pipeline esta parametrizado: cambiar `aeropuerto_oaci` en `conf/base/parameters_retrasos_ml.yml` basta para correr otro aeropuerto. Procesar los 69 en simultaneo requiere mover el parquet a un bucket en la nube y procesarlo por particiones — es un problema de infraestructura, no de metodo.

**5. Segmentacion de rutas en el baseline.** La variable `aeropuerto_dgac_orig_dest` fue excluida por alta cardinalidad, pero con target encoding podria elevar la precision del clasificador de internacionales.

---

## Estructura del proyecto

```
OperacionesAeronaves/
│
├── data/
│   ├── 01_raw/          ← fuentes originales (no versionadas)
│   │   ├── bitacora-vuelos.parquet
│   │   ├── operaciones-aeropuertos.csv
│   │   └── clima_diario_scte.csv      ← clima Meteostat 85799
│   ├── 02_intermediate/ ← datos en transformacion + metricas
│   ├── 03_primary/      ← dato integrado listo para ML
│   │   ├── vuelos_con_operaciones.parquet   (baseline)
│   │   └── vuelos_clima.parquet             (retrasos)
│   ├── 06_models/       ← random_forest.pkl, kmeans.pkl, modelo_retrasos.pkl
│   └── 08_reporting/    ← reportes de inventario
│
├── images/              ← graficos EDA y ML generados automaticamente
│   ├── eda_*.png        ← analisis exploratorio (baseline)
│   ├── ml_*.png         ← clustering y clasificacion (baseline)
│   └── rt_*.png         ← PoC de retrasos
│
├── notebooks/
│   └── informe_ml_operaciones.ipynb   ← informe ejecutable, recorre CRISP-DM
│
├── scripts/
│   └── descargar_clima_scte.py   ← descarga el clima desde Meteostat
│
├── src/operaciones_aeronaves/pipelines/
│   ├── data_inventory/  ← escaneo de archivos raw
│   ├── data_processing/ ← limpieza, JOIN y EDA basico
│   ├── ml/              ← EDA avanzado, clustering y clasificacion
│   └── retrasos_ml/     ← PoC de riesgo de retraso con clima
│
├── tests/
│   ├── test_run.py                  ← humo: los 4 pipelines se registran
│   └── pipelines/retrasos_ml/       ← 16 tests de la reconstruccion horaria
│
├── conf/base/
│   ├── catalog.yml      ← registro de todos los datasets
│   └── parameters*.yml  ← parametros de cada pipeline
│
└── README.md
```
