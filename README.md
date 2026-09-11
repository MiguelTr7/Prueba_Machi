## Modelo_ML_waymo

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Guia de Inicio Rapido

El proyecto usa **uv** como gestor de entornos. No es necesario activar el entorno manualmente.

```bash
# Clonar y entrar
git clone https://github.com/donMixho/OperacionesAeronaves.git
cd OperacionesAeronaves

# Instalar dependencias (solo la primera vez)
uv sync

# Pipeline de preprocesamiento y EDA basico
uv run kedro run --pipeline=data_processing

# Pipeline de ML (EDA avanzado + clustering + clasificacion)
uv run kedro run --pipeline=ml

# Un nodo especifico
uv run kedro run --pipeline=ml --nodes=evaluar_clasificador_node
```

Para activar el venv manualmente (Git Bash / Linux / macOS):
```bash
source .venv/bin/activate
```

---

## Overview

Proyecto de Machine Learning sobre datos de **operaciones aeronauticas de Chile** (JAC - Junta de Aeronautica Civil), publicados en el portal de datos abiertos del Gobierno.

| Fuente | URL |
|---|---|
| Dataset principal | https://datos.gob.cl/dataset/operaciones-aeronaves |
| Documentacion bitacora-vuelos.parquet | https://jac-mtt.github.io/jac-docs/docs/series/operaciones_aeroportuarias/#bitacora-de-aeropuertos |
| Documentacion operaciones-aeropuertos.csv | https://jac-mtt.github.io/jac-docs/docs/series/operaciones_aeroportuarias/#operaciones-por-aeropuertos |

---

## Estado del pipeline

| Etapa | Nodo | Estado | Output |
|---|---|---|---|
| Ingesta | `vuelos_raw`, `aeropuertos_raw` | Completado | 11 074 197 filas |
| Preprocesamiento | `preprocess_vuelos_node` | Completado | `vuelos_preprocessed.parquet` |
| JOIN | `join_con_aeropuertos_node` | Completado | 99.996% cobertura |
| EDA basico | `generate_eda_plots_node` | Completado | 4 imagenes |
| Perfil aeropuertos | `perfil_aeropuertos_node` | Completado | 69 aeropuertos |
| EDA avanzado | `eda_avanzado_node` | Completado | 3 imagenes |
| Clustering K-Means | `cluster_aeropuertos_node` | Completado | 4 clusters |
| Evaluacion clustering | `evaluar_clustering_node` | Completado | Silhouette 0.420 en k=4 |
| Clasificacion RF | `train_clasificador_node` | Completado | ROC-AUC 0.965 |
| Validacion cruzada | `evaluar_clasificador_node` | Completado | CV-5 ROC-AUC 0.963±0.001 |

---

## Pipelines

### `data_processing` — Ingesta y preprocesamiento

**Fuentes de entrada:**

| Dataset | Archivo | Descripcion |
|---|---|---|
| `vuelos_raw` | `data/01_raw/bitacora-vuelos.parquet` | Bitacora de vuelos (11 074 197 filas, 12 columnas) |
| `aeropuertos_raw` | `data/01_raw/operaciones-aeropuertos.csv` | Operaciones agregadas por aeropuerto, mes y tipo |

**Estrategia de preprocesamiento:**

| Columna | Problema | Estrategia |
|---|---|---|
| `numero_vuelo` | 457 687 nulos (4.1%) | Imputacion por categoria fija: "DESCONOCIDO" |
| `pmd` | 275 115 nulos (2.5%) | Flag `pmd_fue_imputado` + mediana por `modelo_avion` (fallback: mediana global) |
| `mes_id` | No existia | Extraido de `dt_operacion` en formato YYYYMM (zona America/Santiago) |
| `internacional_domestico` | No existia | Mapeado desde bool `es_internacional`: True->'I', False->'D' |

**JOIN:** LEFT JOIN sobre `[aeropuerto_oaci, mes_id, internacional_domestico]`. Cobertura: **99.996%** (400 filas sin match de 11 074 197).

---

### `ml` — Modelado y evaluacion

#### Variable objetivo (Target)

**`es_internacional` (binaria)**

Justificacion: es la variable de mayor relevancia operativa y regulatoria del dataset. Distingue el regimen de operacion de cada vuelo (Convenio de Chicago vs. normas nacionales DGAC), determina los requisitos de rampa y aduana, e impacta directamente en la planificacion de capacidad aeroportuaria. La prediccion correcta permite anticipar demanda de infraestructura internacional sin necesidad de datos adicionales.

- Clase negativa (Domestico): 9 654 316 filas (87.2%)
- Clase positiva (Internacional): 1 419 881 filas (12.8%)
- **Desbalance 6.8:1** → se aplica `class_weight="balanced"` en el modelo

#### Features usadas

| Feature | Descripcion | Tipo |
|---|---|---|
| `aeropuerto_oaci` (enc) | Codigo OACI del aeropuerto | Categorica (69 categorias) |
| `actividad_cod` (enc) | Tipo de actividad: U=comercial, P=privado, O=otra... | Categorica (26 categorias) |
| `tipo_operacion` (enc) | A=aterrizaje, D=despegue, W=sobrevuelo | Categorica (3 categorias) |
| `pmd_log` | log(1 + PMD) | Numerica continua |
| `year` | Ano de la operacion (1999-2026) | Numerica discreta |
| `month` | Mes de la operacion (1-12) | Numerica discreta |

---

## Modelado No Supervisado — Clustering de Aeropuertos

**Algoritmo:** K-Means

**Features (con transformacion log + StandardScaler):**
`total_vuelos`, `pct_intl`, `pmd_mediana`, `n_aerolineas`, `cnt_ops_media`

### Seleccion del numero de clusters (k)

![Elbow y Silhouette](images/ml_04_elbow_silhouette.png)

| k | Inercia (WCSS) | Silhouette Score |
|---|---|---|
| 2 | 212.7 | 0.366 |
| 3 | 156.5 | 0.401 |
| **4** | **110.8** | **0.420 (maximo)** |
| 5 | 84.1 | 0.405 |
| 6 | 69.4 | 0.403 |
| 7 | 57.4 | 0.390 |
| 8 | 50.4 | 0.390 |

**k=4 es la eleccion optima**: tiene el mayor Silhouette Score (0.420) y corresponde al punto de inflexion de la curva del codo. A partir de k=5 la ganancia en inercia no justifica la complejidad adicional.

### Resultado del clustering

![Clusters aeropuertos](images/ml_01_clusters_aeropuertos.png)

| Cluster | N aeropuertos | Vuelos totales | % Internacional | PMD mediana | Descripcion operativa |
|---|---|---|---|---|---|
| 0 | 32 | 198 101 | 0.8% | 3.3 t | **Aeropuertos regionales / pista corta** — Operan principalmente aviacion general y vuelos charter internos. Baja frecuencia, aviones livianos. |
| 1 | 16 | 2 892 379 | 3.2% | 68.5 t | **Aeropuertos con operacion pesada** — Combinan lineas comerciales con aviacion de carga y turbopropulsores. Rutas mixtas domestic/internacional de baja frecuencia. |
| 2 | 1 | 3 118 336 | 43.0% | 77.0 t | **Hub internacional dominante (SCEL)** — Unico aeropuerto con volumen comparable a aeropuertos europeos medianos y alta fraccion internacional. Caso singular en el sistema chileno. |
| 3 | 20 | 4 865 381 | 0.3% | 2.0 t | **Grandes hubs domesticos** — Alto volumen de operaciones, pero casi exclusivamente domesticos y con aeronaves livianas (helicópteros, ultralivianos, entrenadores). Incluye SCTB (Tobalaba) y SCIE (Concepcion). |

---

## Modelado Supervisado — Clasificacion RandomForest

**Problema:** Clasificar si un vuelo es internacional (`es_internacional`) a partir de sus caracteristicas operativas.

**Modelo:** RandomForestClassifier
- n_estimators=200, max_depth=12
- class_weight="balanced" (compensa el desbalance 6.8:1)
- n_jobs=-1 (paralelizacion total)

**Datos:** Muestra estratificada de 500 000 filas. Split 80/20 (400 K entrenamiento, 100 K prueba).

### Metricas — Hold-out 20%

| Metrica | Valor | Interpretacion |
|---|---|---|
| **Accuracy** | 0.853 | 85.3% de predicciones correctas |
| **Precision** | 0.465 | 46.5% de los vuelos predichos como internacionales realmente lo son |
| **Recall** | 0.965 | El modelo detecta el 96.5% de los vuelos internacionales reales |
| **F1 Score** | 0.628 | Balance precision-recall |
| **ROC-AUC** | 0.965 | Excelente capacidad discriminatoria general |

### Validacion Cruzada — 5-Fold Estratificado

![Curva ROC](images/ml_05_roc_curve.png)

| Metrica | Media (CV-5) | Desv. Estandar |
|---|---|---|
| Accuracy | 0.852 | ±0.002 |
| Precision | 0.464 | ±0.004 |
| Recall | 0.963 | ±0.006 |
| F1 Score | 0.630 | ±0.006 |
| **ROC-AUC** | **0.964** | **±0.001** |

La baja desviacion estandar en ROC-AUC (±0.001) confirma que el modelo no presenta sobreajuste y generaliza de forma consistente entre folds.

### Importancia de features

![Importancia features](images/ml_02_importancia_features.png)

El **aeropuerto** es la variable mas predictiva, lo que tiene sentido operativo: solo ciertos aeropuertos tienen rutas internacionales. El **PMD** es el segundo predictor: los aviones mas pesados se destinan preferentemente a rutas internacionales de largo alcance.

### Matriz de confusion

![Confusion matrix](images/ml_03_confusion_matrix.png)

Con `class_weight=balanced` el modelo maximiza el Recall (96.5%): captura casi todos los vuelos internacionales a costa de clasificar algunos vuelos domesticos como internacionales (falsos positivos). Esta configuracion es apropiada cuando el costo de no detectar un vuelo internacional es mayor que el de un falso positivo.

---

## Conclusiones de negocio

### 1. SCEL es un sistema en si mismo
Santiago opera como un hub de escala regional (no solo nacional), con un perfil operativo completamente diferente al resto del sistema aeroportuario chileno. Cualquier modelo que incluya SCEL debe tratarlo como una entidad separada o usar variables de interaccion especificas.

### 2. El cluster de hubs domesticos (Cluster 3) tiene potencial subestimado
Aeropuertos como SCTB (Tobalaba, 1.3M operaciones) operan principalmente aviacion general y entrenamiento. Su alto volumen refleja intensidad de uso, no trafico comercial, lo que los hace candidatos a regulacion diferenciada.

### 3. La imputacion por mediana grupal es robusta
La distribucion de PMD imputado es estadisticamente indistinguible de la distribucion original, y el modelo no usa `pmd_fue_imputado` como feature relevante, confirmando que la imputacion no introduce sesgo.

### 4. El modelo de clasificacion es util para priorizar infraestructura
Con ROC-AUC de 0.964 (CV-5), el clasificador puede aplicarse a registros futuros para estimar la probabilidad de que una operacion sea internacional, permitiendo planificar:
- Necesidades de despacho de aduana y migracion
- Capacidad de gate internacional
- Asignacion de slots en aeropuertos congestionados

### 5. Siguiente paso recomendado
Incorporar como feature la **ruta origen-destino** (`aeropuerto_dgac_orig_dest`) para mejorar la precision del modelo sin sacrificar recall. Actualmente excluida por cardinalidad alta, pero con target encoding podria aportar informacion critica sobre pares de rutas internacionales historicamente establecidos.

---

## Hallazgos del EDA

### Top 15 aeropuertos por volumen
![Top 15](images/eda_01_top_aeropuertos.png)

### Internacional vs Domestico por ano
![Intl vs Dom](images/eda_02_intl_vs_dom.png)

### PMD imputado vs original
![PMD](images/eda_03_pmd_imputado.png)

### Evolucion temporal 1999-2026
![Evolucion](images/eda_04_operaciones_anio.png)

### PMD mediana vs % internacional por aeropuerto
![PMD vs Intl](images/eda_05_pmd_vs_intl_aeropuerto.png)

### Heatmap Top-12 aeropuertos x ano
![Heatmap](images/eda_06_heatmap_top12.png)

### Correlacion entre variables numericas
![Correlacion](images/eda_07_correlacion_numericas.png)

---

## Estructura del proyecto

```
OperacionesAeronaves/
├── conf/base/
│   ├── catalog.yml                    # todos los datasets
│   ├── parameters.yml
│   └── parameters_data_inventory.yml
├── data/
│   ├── 01_raw/
│   │   ├── bitacora-vuelos.parquet    # no versionado
│   │   └── operaciones-aeropuertos.csv
│   └── 02_intermediate/               # parquets de salida de cada pipeline
│       ├── vuelos_preprocessed.parquet
│       ├── vuelos_con_operaciones.parquet
│       ├── aeropuertos_perfil.parquet
│       ├── cluster_labels.parquet
│       ├── cluster_metricas.parquet   # silhouette/inercia por k
│       ├── metricas_supervisado.parquet
│       └── metricas_cv_supervisado.parquet
├── images/                            # graficos EDA y ML
│   ├── eda_01..07_*.png
│   ├── ml_01_clusters_aeropuertos.png
│   ├── ml_02_importancia_features.png
│   ├── ml_03_confusion_matrix.png
│   ├── ml_04_elbow_silhouette.png
│   └── ml_05_roc_curve.png
├── src/modelo_ml_waymo/pipelines/
│   ├── data_inventory/
│   ├── data_processing/
│   │   ├── nodes.py
│   │   └── pipeline.py
│   └── ml/
│       ├── nodes.py
│       └── pipeline.py
└── README.md
```
