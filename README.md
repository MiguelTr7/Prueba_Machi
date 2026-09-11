# Modelo ML — Operaciones Aeronauticas Chile (JAC)

> Proyecto de Machine Learning sobre la bitacora de vuelos de Chile (1999–2026).
> Fuente: [datos.gob.cl — Junta de Aeronautica Civil](https://datos.gob.cl/dataset/operaciones-aeronaves)

---

## Indice

- [¿De que trata este proyecto?](#de-que-trata-este-proyecto)
- [Arquitectura y Pipeline](#arquitectura-y-pipeline)
- [Como ejecutar el proyecto](#como-ejecutar-el-proyecto)
- [Preparacion de los datos](#preparacion-de-los-datos)
- [Analisis Exploratorio — Graficos clave](#analisis-exploratorio--graficos-clave)
- [Modelado No Supervisado — Clustering](#modelado-no-supervisado--clustering)
- [Modelado Supervisado — Clasificacion](#modelado-supervisado--clasificacion)
- [Conclusiones y Decisiones de Negocio](#conclusiones-y-decisiones-de-negocio)
- [Estructura del proyecto](#estructura-del-proyecto)

---

## ¿De que trata este proyecto?

Chile tiene 69 aeropuertos activos y registra mas de **11 millones de operaciones aereas** entre 1999 y 2026. Esta base de datos — publicada abiertamente por la JAC — es una de las series historicas mas completas de transporte aereo en Latinoamerica.

Este proyecto responde tres preguntas concretas:

- **¿Como se distribuye el trafico aereo en Chile?** ¿Que aeropuertos concentran el poder? ¿Como cambia eso en el tiempo?
- **¿Se pueden agrupar los aeropuertos por su comportamiento operativo?** ¿Hay patrones que el dato "bruto" no deja ver?
- **¿Se puede predecir si un vuelo es internacional o domestico** solo con sus datos operativos (aeropuerto, avion, mes)?

La respuesta a las tres es sí — y este proyecto lo demuestra con datos.

---

## Arquitectura y Pipeline

El proyecto esta construido con **Kedro**, un framework que convierte cada transformacion en un nodo trazable y reproducible. El flujo completo tiene **12 nodos** que corren en ~82 segundos.

```
vuelos_raw (11M filas)
    │
    ▼
preprocess_vuelos          ← limpieza + variables nuevas
    │
    ▼
join_con_aeropuertos       ← une con estadisticas mensuales por aeropuerto
    │ (03_primary)
    ├──▶ generate_eda_plots     ← 4 graficos basicos
    ├──▶ perfil_aeropuertos     ── cluster_aeropuertos ─▶ kmeans.pkl
    │                           └─ evaluar_clustering  ─▶ curva del codo
    ├──▶ eda_avanzado           ← 3 graficos de correlacion y distribucion
    ├──▶ train_clasificador     ─▶ random_forest.pkl + metricas
    └──▶ evaluar_clasificador   ─▶ CV-5 + curva ROC
```

**Pipelines registrados:**

| Pipeline | Que hace |
|---|---|
| `data_inventory` | Inventario de archivos en `data/01_raw/` |
| `data_processing` | Limpieza, JOIN y EDA basico |
| `ml` | EDA avanzado, clustering y clasificacion |

---

## Como ejecutar el proyecto

```bash
# Clonar e instalar
git clone https://github.com/donMixho/OperacionesAeronaves.git
cd OperacionesAeronaves
uv sync

# Ejecutar todo de una vez (~82 segundos)
uv run kedro run

# O por partes
uv run kedro run --pipeline=data_processing
uv run kedro run --pipeline=ml
```

> **Semilla fija:** todos los modelos usan `random_state=42` para garantizar resultados identicos entre ejecuciones.

---

## Preparacion de los datos

Las dos fuentes crudas tienen problemas tipicos de datos reales. Asi los resolvimos:

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

**¿Que descubrimos?** Las dos curvas — valores originales e imputados — son casi identicas. El pequeño "pico" que aparece en los valores imputados es normal: cuando se imputa con una mediana, varios registros reciben exactamente el mismo valor (el valor central del grupo), lo que crea una concentracion visible pero que **no distorsiona la distribucion real**. La fisica del avion no cambia — simplemente usamos la mejor estimacion disponible para ese modelo.

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

**¿Que descubrimos?** El **aeropuerto** es el predictor dominante — tiene sentido: solo ciertos aeropuertos tienen rutas internacionales. El **PMD** (peso del avion) es el segundo predictor: los aviones mas pesados casi siempre vuelan mas lejos. El mes y el año tambien aportan — el trafico internacional tiene estacionalidad y ha crecido con los anos.

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

## Conclusiones y Decisiones de Negocio

**1. El sistema aereo chileno es un monocentro.**
SCEL concentra el trafico internacional de forma casi monopolica. Cualquier decision de politica de expansion de rutas internacionales pasa obligatoriamente por Santiago — no hay alternativa real en el corto plazo.

**2. Los aeropuertos del Cluster 3 son los grandes olvidados del analisis tradicional.**
Aeropuertos como Tobalaba (SCTB) o Concepcion (SCIE) tienen volumenes altisimos de operaciones, pero casi toda es aviacion general o entrenamiento. No son "grandes" por tener muchos pasajeros — son grandes por intensidad de uso. Requieren regulacion diferenciada.

**3. La imputacion de PMD es confiable.**
El 2.5% de registros con PMD faltante fue imputado con la mediana del modelo de avion. La distribucion resultante es indistinguible de la original. Este dato puede usarse con confianza en futuros modelos.

**4. El modelo puede operar en produccion.**
Con ROC-AUC de 0.964 estable en CV-5, el clasificador esta listo para aplicarse a registros nuevos. El modelo serializado esta en `data/06_models/random_forest.pkl` y se puede cargar con `joblib.load()` sin re-entrenar.

**5. El siguiente paso natural es la segmentacion de rutas.**
La variable `aeropuerto_dgac_orig_dest` (destino del vuelo) fue excluida por alta cardinalidad, pero con target encoding podria elevar la precision del modelo de forma significativa — especialmente para rutas internacionales poco frecuentes que el modelo actual confunde con domesticas.

---

## Estructura del proyecto

```
OperacionesAeronaves/
│
├── data/
│   ├── 01_raw/          ← fuentes originales (no versionadas)
│   ├── 02_intermediate/ ← datos en transformacion
│   ├── 03_primary/      ← dato integrado listo para ML (vuelos_con_operaciones.parquet)
│   ├── 06_models/       ← modelos entrenados (random_forest.pkl, kmeans.pkl)
│   └── 08_reporting/    ← reportes de inventario
│
├── images/              ← graficos EDA y ML generados automaticamente
│
├── src/modelo_ml_waymo/pipelines/
│   ├── data_inventory/  ← escaneo de archivos raw
│   ├── data_processing/ ← limpieza, JOIN y EDA basico
│   └── ml/              ← EDA avanzado, clustering y clasificacion
│
├── conf/base/
│   ├── catalog.yml      ← registro de todos los datasets
│   └── parameters*.yml  ← parametros de cada pipeline
│
└── README.md
```
