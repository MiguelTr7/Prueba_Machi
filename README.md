## Modelo_ML_waymo

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## 🚀 Guía de Inicio Rápido (Git Bash)

Sigue estos pasos desde la terminal de **Git Bash** para levantar y ejecutar el proyecto en cualquier computadora desde cero:

### 1. Clonar el repositorio y entrar al proyecto
git clone <https://github.com/donMixho/Modelo_ML_waymo.git>
cd Modelo_ML_waymo

### 2. Crear y activar el entorno virtual
python -m venv .venv

source .venv/Scripts/activate

*(Nota: Al activarse correctamente verás (.venv) al inicio de tu línea de comandos).*

### 3. Instalar dependencias
pip install -r requirements.txt

### 4. Ejecutar los pipelines de Kedro
Para ejecutar todos los pipelines del proyecto a la vez:
kedro run

O si prefieres ejecutar un pipeline individual:
kedro run --pipeline=data_inventory
kedro run --pipeline=eda
kedro run --pipeline=image_ingestion

---

## Overview

Proyecto Kedro (`kedro 1.3.1`) para el análisis exploratorio y la verificación de señal
predictiva sobre un dataset sintético de detecciones estilo Waymo:
[data/01_raw/detecciones_waymo_like.csv](data/01_raw/detecciones_waymo_like.csv).

**Importante:** este CSV no es el Waymo Open Dataset real. Es una tabla plana de
detecciones (153 segmentos, ~40,680 filas) con cajas delimitadoras 3D, sin imágenes,
cámaras ni tfrecords. Todo el trabajo hecho hasta ahora es de **diagnóstico**: inventario,
perfilado, auditoría de calidad y verificación de señal. No se ha limpiado, imputado ni
transformado ningún dato de forma persistente.

## Pipelines

### `data_inventory`

Inventario de solo lectura de `data/01_raw`, sin asumir ningún formato previo.

| Nodo | Qué hace |
| --- | --- |
| `scan_raw_files` | Recorre `data/01_raw` y genera un índice de archivos (ruta, formato, tamaño). |
| `profile_sample` | Abre una muestra pequeña y detecta el esquema disponible (columnas, tipos de anotación, metadata). |
| `build_inventory_report` | Consolida ambos en una tabla resumen y un reporte Markdown. |

Salidas: `data/02_intermediate/data_inventory_*.parquet` y
[data/08_reporting/data_inventory_report.md](data/08_reporting/data_inventory_report.md).

### `eda`

Análisis exploratorio completo + verificación de señal predictiva.

| Nodo | Qué hace |
| --- | --- |
| `load_and_profile` | Perfil por columna: dtype, nulos, cardinalidad, estadísticos o top-10 de valores. |
| `audit_categorical_consistency` | Lista los valores crudos de las columnas categóricas y **propone** (sin aplicar) una normalización canónica. |
| `audit_data_quality` | Duplicados, valores físicamente imposibles, sentinels no numéricos, outliers (IQR/z-score), co-ocurrencia de nulos y consistencia temporal por segmento. |
| `target_candidates_analysis` | Balance de clases, tablas cruzadas y asociación (Cramér's V) de `detection_difficulty` y `object_type` contra weather/time_of_day/object_type. |
| `generate_plots` | Histogramas, boxplots, barplots, matriz de correlación y heatmap de nulos (PNG). |
| `build_eda_report` | Consolida todo en un reporte Markdown en español. |
| `signal_check` | Para cada candidato a target: normalización en memoria, información mutua (con ruido de referencia), test de separabilidad (Kruskal-Wallis) y un baseline honesto (dummy vs. árbol de decisión) con split estratificado y deduplicado. Veredicto: SEÑAL / SEÑAL DÉBIL / SIN SEÑAL. |

Salidas: `data/02_intermediate/eda_*.parquet` y `data/02_intermediate/signal_*.parquet`,
figuras en `data/08_reporting/figures/`, y dos reportes:
[data/08_reporting/eda_report.md](data/08_reporting/eda_report.md) y
[data/08_reporting/signal_check_report.md](data/08_reporting/signal_check_report.md).

Todos los nodos son de solo lectura sobre `data/01_raw`: ninguno imputa, elimina filas ni
normaliza datos de forma persistente. Las normalizaciones y coerciones numéricas que se ven
en el código (p. ej. `object_type` en `signal_check`, o el sentinel `"N/D"` en
`timestamp_micros`) existen solo en memoria, dentro de la función que las necesita.

### `image_ingestion`

Extracción de imágenes de cámara desde tfrecords reales del Waymo Open Dataset
**PERCEPTION** (Frame proto) ya descargados localmente en
`data/01_raw/perception_tfrecords/`. Este pipeline no descarga nada por sí solo: solo lee lo
que ya está en disco.

| Nodo | Qué hace |
| --- | --- |
| `list_perception_tfrecords` | Lista los `.tfrecord` en `data/01_raw/perception_tfrecords`. Si el directorio no existe o está vacío, emite un warning y no falla. |
| `extract_and_grayscale` | Para cada tfrecord, parsea cada `Frame`, decodifica cada `frame.images[i]` con OpenCV y lo guarda como PNG en escala de grises en `data/02_raw/` (`{context_name}_{timestamp_micros}_{camera_name}.png`). Es la única transformación: sin resize, crop ni normalización. |
| `build_image_metadata_table` | Consolida, por cada imagen guardada, `image_path`, `context_name`, `timestamp_micros`, `camera_name`, `time_of_day` y `weather` (tal como vienen en `frame.context.stats`, sin imputar nulos). |

Salidas: `data/02_raw/*.png` (imágenes), `data/02_raw/image_metadata.parquet` y los
intermedios `data/02_intermediate/image_ingestion_tfrecord_index.parquet` /
`data/02_intermediate/image_ingestion_records.json`.

**Paso previo obligatorio (una sola vez):** el paquete `Frame`/`CameraImage`/`CameraName`
normalmente vendría del pip `waymo-open-dataset-tf-2-12-0`, pero ese paquete no publica wheel
para Windows y no instala. En su lugar, este proyecto vendoriza solo los `.proto` que
necesita (`src/modelo_ml_waymo/waymo_protos_src/`) y los compila localmente con
`grpcio-tools` (protobuf puro, sí instala en Windows). Después de `pip install -r
requirements.txt` (o `uv sync`) y **antes** de correr `kedro run --pipeline=image_ingestion`,
hay que generar los módulos `_pb2.py` una vez:

```
python scripts/compile_waymo_protos.py
```

Esto genera `src/waymo_open_dataset/dataset_pb2.py` y el resto de los `_pb2.py` a partir de
los `.proto` fuente. Se generan en `src/waymo_open_dataset/` (nivel superior, hermano de
`src/modelo_ml_waymo/`) y no dentro del paquete `modelo_ml_waymo`, porque `dataset_pb2.py`
usa imports absolutos tipo `from waymo_open_dataset import label_pb2`, que requieren que
`waymo_open_dataset` sea un paquete de nivel superior visible en `sys.path` (Kedro ya agrega
`src/` a `sys.path`). Es seguro volver a correr el script si los `.proto` cambian;
simplemente sobrescribe los módulos generados.

## Estructura del proyecto

```
modelo-ml-waymo/
├── conf/
│   ├── base/
│   │   ├── catalog.yml                  # datasets de los tres pipelines
│   │   ├── parameters.yml
│   │   ├── parameters_data_inventory.yml
│   │   ├── parameters_eda.yml           # incluye parámetros de signal_check
│   │   └── parameters_image_ingestion.yml
│   └── local/                           # credenciales/config local (no se versiona)
├── data/
│   ├── 01_raw/
│   │   ├── detecciones_waymo_like.csv   # dataset fuente (no versionado, ver .gitignore)
│   │   └── perception_tfrecords/        # tfrecords reales del Waymo Open Dataset PERCEPTION
│   ├── 02_raw/                          # PNG en escala de grises + image_metadata.parquet
│   ├── 02_intermediate/                 # salidas parquet/json de los tres pipelines
│   └── 08_reporting/
│       ├── data_inventory_report.md
│       ├── eda_report.md
│       ├── signal_check_report.md
│       └── figures/                     # PNG generados por generate_plots
├── src/modelo_ml_waymo/
│   ├── pipelines/
│   │   ├── data_inventory/
│   │   │   ├── nodes.py
│   │   │   └── pipeline.py
│   │   ├── eda/
│   │   │   ├── nodes.py                 # incluye signal_check
│   │   │   └── pipeline.py
│   │   └── image_ingestion/
│   │       ├── nodes.py
│   │       └── pipeline.py
│   ├── waymo_protos_src/                # .proto fuente vendorizados (dataset/label/map/vector/keypoint)
│   ├── pipeline_registry.py             # autodescubre los pipelines de arriba
│   └── settings.py
├── src/waymo_open_dataset/              # _pb2.py generados por scripts/compile_waymo_protos.py (no versionado)
│   └── protos/
├── tests/pipelines/                     # boilerplate de test por pipeline
├── pyproject.toml / requirements.txt / uv.lock
└── README.md
```
