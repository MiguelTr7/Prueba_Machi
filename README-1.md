# Modelo_ML_waymo

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

---

## 📋 Contexto de Evaluación (EP1 — 30% de la nota final)

> Este proyecto corresponde a la **Evaluación Parcial N°1** de la asignatura *Machine Learning* (MLY1101).
> El entregable es un informe técnico en Markdown que documenta todas las etapas del proceso de ciencia de datos
> junto con un notebook ejecutable, siguiendo la metodología **CRISP-DM**.

| Indicador | Peso | ¿Cubierto en este proyecto? |
|---|---|---|
| IE1: Fuentes de datos y herramientas colaborativas | 10% | ✅ Waymo Open Dataset v2.0.1 + GitHub + Kedro + Colab |
| IE2: Manipulación y preparación de datos en Python | 30% | ✅ Pipeline `waymo_v2` (Pandas + PySpark) |
| IE3: Análisis exploratorio y calidad de datos (EDA) | 40% | ✅ Pipelines `data_inventory` + `eda` + `signal_check` |
| IE4: Evaluación de sesgos, ética y privacidad | 20% | ⚠️ Pendiente — ver sección al final |

**Metodología aplicada:** CRISP-DM
1. Comprensión del problema de negocio → [sección "Problema"](#el-problema-que-resolvemos)
2. Comprensión de los datos → pipeline `data_inventory`
3. Preparación de datos → pipeline `waymo_v2` + `eda`
4. Modelamiento → en desarrollo (pipeline ML pendiente)
5. Evaluación → verificación de señal (`signal_check`)
6. Despliegue → pendiente

**Entregables requeridos:**
- [x] Informe técnico Markdown (este README + reportes en `data/08_reporting/`)
- [x] Dataset organizado (ver estructura `data/`)
- [x] Carpeta profesional con `data/`, `notebooks/`, `src/`, `conf/`
- [ ] Notebook Python ejecutable (pendiente — exportar pipeline a `.ipynb`)

---

## El Problema que Resolvemos

### Contexto: vehículos autónomos y fusión de sensores

Un vehículo autónomo como los de Waymo usa múltiples sensores simultáneamente: **LiDAR** (láser que mide distancias en 3D) y **cámaras** (que capturan imagen visual). Cada sensor detecta objetos del entorno de forma independiente.

El desafío real es la **fusión de sensores**: cuando el LiDAR "ve" un objeto (persona, auto, señal), ¿la cámara también lo detecta? Si el LiDAR detecta algo que la cámara no ve, puede ser ruido, un objeto fuera del campo visual, o un fallo del sistema.

### Variable objetivo: `tiene_camara`

**Pregunta:** *dado un objeto detectado por LiDAR (caja 3D con posición, tamaño y tipo), ¿existe una detección de cámara asociada a ese mismo objeto en ese instante?*

- `tiene_camara = 1` → el objeto LiDAR tiene correspondencia en cámara (fusión exitosa)
- `tiene_camara = 0` → el objeto LiDAR no tiene detección de cámara (solo LiDAR)

Esto es un problema de **clasificación binaria**. Un modelo que prediga bien `tiene_camara` puede ayudar a:
- Detectar fallos en la fusión de sensores antes de que afecten la navegación
- Priorizar qué objetos necesitan revisión humana
- Evaluar si ciertas condiciones (lluvia, noche) degradan la cobertura de cámara

### ¿Qué papel juegan las imágenes?

Las imágenes de cámara (JPEG en los datos Waymo) son una fuente de señal visual complementaria al LiDAR. En este proyecto las imágenes se procesan así:

1. **Se convierten a escala de grises** con OpenCV (`cv2.imdecode + IMREAD_GRAYSCALE`)
   - Razón: reduce el espacio de almacenamiento en ~3x (3 canales RGB → 1 canal gris)
   - Razón: reduce el cómputo en la fase de modelamiento futura
   - La información estructural (bordes, formas, siluetas) se conserva íntegramente
2. **Se guardan como PNG** en `data/02_raw/grayscale_images/`
3. **Se genera una tabla de metadata** (`waymo_v2_image_metadata.parquet`) con path, contexto, timestamp y condiciones (clima, hora del día)

En la fase de modelamiento, la metadata de imagen puede usarse como feature (p.ej. ¿hay imagen disponible de esa cámara en ese instante?) o las imágenes en gris pueden alimentar una red neuronal convolucional (CNN) como extractor de features visuales. Por ahora el pipeline solo hace la ingesta y conversión.

### Datos utilizados

**Waymo Open Dataset v2.0.1** — 4 segmentos de conducción seleccionados para representar condiciones contrastantes:

| Segmento | Clima | Hora | Ciudad | % tiene_camara |
|---|---|---|---|---|
| `10206293520369375008` | Soleado | Noche | Phoenix | 10.1% |
| `11017034898130016754` | Soleado | Amanecer/Atardecer | Otra | 6.1% |
| `6791933003490312185` | Lluvia | Amanecer/Atardecer | Phoenix | 2.4% |
| `10023947602400723454` | Soleado | Día | San Francisco | 32.4% |

La variación intencional en condiciones permite evaluar si el modelo generaliza o depende del clima/horario.

---

## KPIs del Proyecto

| KPI | Descripción | Meta |
|---|---|---|
| F1-score (clase positiva) | Capacidad de detectar fusiones exitosas | > 0.70 |
| Accuracy balanceada | Rendimiento considerando desbalance de clases | > 0.65 |
| Cobertura de imagen | % de frames con al menos 1 imagen procesada | 100% |
| Ratio de imbalance | Positivos / Total por segmento | Documentado por segmento |

---

## 🚀 Guía de Inicio Rápido (Git Bash)

### 1. Clonar el repositorio
```bash
git clone https://github.com/donMixho/Modelo_ML_waymo.git
cd Modelo_ML_waymo
```

### 2. Crear y activar el entorno virtual
```bash
python -m venv .venv
source .venv/Scripts/activate
# Debe aparecer (.venv) al inicio de la línea
```

### 3. Instalar dependencias
```bash
python -m pip install -r requirements.txt
python -m pip install "pyspark>=3.5"   # requerido por el pipeline waymo_v2
```

### 4. Colocar los datos de Waymo

Descarga los Parquet desde Google Cloud (ver Colab `MLPruebas1.ipynb`) y colócalos así:
```
data/01_raw/waymo_v2/
├── lidar_box/                    ← archivos lidar_box_*.parquet
├── camera_to_lidar_box_association/  ← archivos assoc_*.parquet
├── stats/                        ← archivos stats_*.parquet
└── camera_image/                 ← archivos camera_image_*.parquet
```

### 5. Ejecutar pipelines
```bash
kedro run --pipeline=data_inventory   # inventario de datos crudos
kedro run --pipeline=eda              # EDA + verificación de señal
kedro run --pipeline=waymo_v2         # procesamiento de datos Waymo reales
```

---

## Verificar que todo está en orden

Antes de ejecutar, corre estos comandos para diagnosticar el estado del proyecto:

```bash
# 1. Listar todos los pipelines registrados (debe aparecer waymo_v2)
kedro registry list

# 2. Listar todos los datasets del catálogo (verifica que los nuevos estén)
kedro catalog list

# 3. Verificar que el entorno tiene las dependencias críticas
python -c "import pandas; print('pandas OK')"
python -c "import cv2; print('opencv OK')"
python -c "import pyspark; print('pyspark OK')"
python -c "from modelo_ml_waymo.pipelines.waymo_v2 import create_pipeline; print('pipeline OK')"

# 4. Ver el grafo del pipeline sin ejecutar nada (requiere kedro-viz)
kedro viz run   # abre en el navegador
# o si no tienes viz instalado:
python -m pip install kedro-viz
```

Si `kedro registry list` no muestra `waymo_v2`, revisa que existan estos archivos:
```
src/modelo_ml_waymo/pipelines/waymo_v2/__init__.py
src/modelo_ml_waymo/pipelines/waymo_v2/pipeline.py
src/modelo_ml_waymo/pipelines/waymo_v2/nodes.py
conf/base/parameters_waymo_v2.yml
```

---

## Pipelines

### `data_inventory`

Inventario de solo lectura de `data/01_raw`, sin asumir ningún formato previo.

| Nodo | Qué hace |
|---|---|
| `scan_raw_files` | Recorre `data/01_raw` y genera un índice de archivos (ruta, formato, tamaño). |
| `profile_sample` | Abre una muestra pequeña y detecta el esquema disponible (columnas, tipos de anotación, metadata). |
| `build_inventory_report` | Consolida ambos en una tabla resumen y un reporte Markdown. |

Salidas: `data/02_intermediate/data_inventory_*.parquet` y `data/08_reporting/data_inventory_report.md`.

### `eda`

Análisis exploratorio completo + verificación de señal predictiva sobre el CSV sintético.

| Nodo | Qué hace |
|---|---|
| `load_and_profile` | Perfil por columna: dtype, nulos, cardinalidad, estadísticos o top-10 de valores. |
| `audit_categorical_consistency` | Lista los valores crudos de las columnas categóricas y propone una normalización canónica. |
| `audit_data_quality` | Duplicados, valores físicamente imposibles, sentinels no numéricos, outliers (IQR/z-score), co-ocurrencia de nulos y consistencia temporal por segmento. |
| `target_candidates_analysis` | Balance de clases, tablas cruzadas y asociación (Cramér's V) de `detection_difficulty` y `object_type` contra weather/time_of_day/object_type. |
| `generate_plots` | Histogramas, boxplots, barplots, matriz de correlación y heatmap de nulos (PNG). |
| `build_eda_report` | Consolida todo en un reporte Markdown en español. |
| `signal_check` | Información mutua, test Kruskal-Wallis y baseline (dummy vs. árbol de decisión). Veredicto: SEÑAL / SEÑAL DÉBIL / SIN SEÑAL. |

**Resultado del signal_check:** `object_type` tiene señal brutal (F1 árbol=0.9963 vs dummy=0.1909). Las features más importantes son `box_length`, `box_width`, `speed_mps`.

Salidas: `data/02_intermediate/eda_*.parquet`, figuras en `data/08_reporting/figures/`, y reportes `eda_report.md` y `signal_check_report.md`.

### `image_ingestion` *(legacy — modo tfrecords)*

Pipeline original para tfrecords del Waymo PERCEPTION. **Obsoleto para Waymo v2.0.1** (que usa Parquet, no tfrecords). Se mantiene en el repo como referencia. Usar `waymo_v2` para datos reales.

### `waymo_v2` *(pipeline principal — datos reales)*

Procesamiento del Waymo Open Dataset v2.0.1 en formato Parquet. Requiere los archivos en `data/01_raw/waymo_v2/`.

| Nodo | Qué hace |
|---|---|
| `load_and_merge_parquets` | Lee lidar_box + association + stats, crea `tiene_camara` por merge eficiente, une metadata de condiciones (clima, hora, ciudad). |
| `extract_grayscale_images` | Decodifica imágenes JPEG de cámara con OpenCV, guarda PNG en escala de grises en `data/02_raw/grayscale_images/`. |
| `build_spark_feature_table` | Aplica transformaciones con PySpark (`object_type_label` int→string, `box_volume` = x·y·z). Tiene fallback pandas si PySpark no está instalado. |
| `build_waymo_v2_report` | Reporte Markdown con distribución de target, condiciones, tipos de objeto y estadísticos de volumen. |

Salidas: `waymo_v2_features.parquet`, `waymo_v2_image_metadata.parquet`, `waymo_v2_report.md`.

---

## ⚠️ IE4: Sesgos, Ética y Privacidad *(pendiente)*

Este apartado debe completarse antes de la evaluación. Puntos a cubrir:

- **Sesgo de selección:** los 4 segmentos elegidos sobrerrepresentan Phoenix (2 de 4). ¿Generaliza a otras ciudades?
- **Sesgo de clase:** `tiene_camara` varía de 2.4% a 32.4% según el segmento — el modelo podría aprender a separar segmentos en vez de la variable real.
- **Sesgo temporal:** lluvia de noche vs. sol de día están correlacionados con `tiene_camara`. ¿Es el clima la causa real o es un confounder?
- **Privacidad:** las imágenes de Waymo pueden contener rostros de personas y patentes de vehículos. Waymo ya aplica anonimización en el dataset público, pero debe documentarse.
- **Impacto:** un modelo que falle sistemáticamente en lluvia podría afectar la seguridad de personas en condiciones adversas.

---

## Estructura del proyecto

```
modelo-ml-waymo/
├── conf/base/
│   ├── catalog.yml
│   ├── parameters_waymo_v2.yml
│   ├── parameters_data_inventory.yml
│   ├── parameters_eda.yml
│   └── parameters_image_ingestion.yml
├── data/
│   ├── 01_raw/
│   │   ├── detecciones_waymo_like.csv   # CSV sintético (no versionado)
│   │   └── waymo_v2/                    # Parquet reales (no versionados)
│   │       ├── lidar_box/
│   │       ├── camera_to_lidar_box_association/
│   │       ├── stats/
│   │       └── camera_image/
│   ├── 02_raw/
│   │   └── grayscale_images/            # PNG en escala de grises
│   ├── 02_intermediate/                 # Parquet intermedios de todos los pipelines
│   └── 08_reporting/
│       ├── data_inventory_report.md
│       ├── eda_report.md
│       ├── signal_check_report.md
│       ├── waymo_v2_report.md
│       └── figures/
├── src/modelo_ml_waymo/
│   ├── pipelines/
│   │   ├── data_inventory/
│   │   ├── eda/
│   │   ├── image_ingestion/             # legacy
│   │   └── waymo_v2/                   # pipeline principal
│   ├── waymo_protos_src/               # .proto vendorizados (legacy)
│   └── pipeline_registry.py
├── notebooks/                           # pendiente: exportar pipelines a .ipynb
├── scripts/
│   └── compile_waymo_protos.py
├── requirements.txt
└── README.md
```
