"""
Pipeline 'waymo_v2'
Procesamiento del Waymo Open Dataset v2.0.1 (formato Parquet).
Lee lidar_box, camera_to_lidar_box_association, stats y camera_image,
crea el target 'tiene_camara', convierte imagenes a escala de grises con OpenCV
y aplica transformaciones representativas con PySpark.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_IMAGE_METADATA_COLUMNS = [
    "image_path",
    "context_name",
    "timestamp_micros",
    "camera_name",
    "time_of_day",
    "weather",
]

# Mapeo de int -> nombre legible para [LiDARBoxComponent].type
_LIDAR_TYPE_MAP = {
    1: "TYPE_VEHICLE",
    2: "TYPE_PEDESTRIAN",
    3: "TYPE_SIGN",
    4: "TYPE_CYCLIST",
}


# ---------------------------------------------------------------------------
# 1. load_and_merge_parquets
# ---------------------------------------------------------------------------


def load_and_merge_parquets(
    lidar_box_dir: str,
    assoc_dir: str,
    stats_dir: str,
) -> pd.DataFrame:
    """Lee los parquets de lidar_box, camera_to_lidar_box_association y stats,
    crea la columna target ``tiene_camara`` y une los metadatos de contexto.

    Ninguna fila se elimina ni imputa. La columna ``tiene_camara`` es 1 si el
    objeto LiDAR tiene correspondencia en al menos una camara en ese frame,
    0 en caso contrario.

    Args:
        lidar_box_dir: Directorio con los parquets de lidar_box.
        assoc_dir:     Directorio con los parquets de camera_to_lidar_box_association.
        stats_dir:     Directorio con los parquets de stats (time_of_day, weather).

    Returns:
        DataFrame con features LiDAR, target ``tiene_camara`` y metadatos de contexto.
    """
    # ---- lidar_box --------------------------------------------------------
    lidar_files = sorted(Path(lidar_box_dir).glob("*.parquet"))
    if not lidar_files:
        logger.warning("No se encontraron parquets en %s.", lidar_box_dir)
        return pd.DataFrame()

    df_lidar = pd.concat(
        [pd.read_parquet(f) for f in lidar_files], ignore_index=True
    )
    logger.info("lidar_box: %d filas, %d segmentos.", len(df_lidar), df_lidar["key.segment_context_name"].nunique())

    # ---- camera_to_lidar_box_association ----------------------------------
    assoc_files = sorted(Path(assoc_dir).glob("*.parquet"))
    if assoc_files:
        df_assoc = pd.concat(
            [pd.read_parquet(f) for f in assoc_files], ignore_index=True
        )
        # Subconjunto unico de (segmento, frame, laser_object_id) con camara
        assoc_keys = (
            df_assoc[
                [
                    "key.segment_context_name",
                    "key.frame_timestamp_micros",
                    "key.laser_object_id",
                ]
            ]
            .drop_duplicates()
            .assign(tiene_camara=1)
        )
        df_lidar = df_lidar.merge(
            assoc_keys,
            on=[
                "key.segment_context_name",
                "key.frame_timestamp_micros",
                "key.laser_object_id",
            ],
            how="left",
        )
        df_lidar["tiene_camara"] = df_lidar["tiene_camara"].fillna(0).astype(int)
        pct = df_lidar["tiene_camara"].mean() * 100
        logger.info(
            "Target 'tiene_camara': %d positivos (%.1f%%) de %d totales.",
            df_lidar["tiene_camara"].sum(),
            pct,
            len(df_lidar),
        )
    else:
        logger.warning(
            "No se encontraron parquets en %s; 'tiene_camara' sera NaN.", assoc_dir
        )
        df_lidar["tiene_camara"] = None

    # ---- stats (time_of_day, weather, location) ---------------------------
    stats_files = sorted(Path(stats_dir).glob("*.parquet"))
    if stats_files:
        df_stats_raw = pd.concat(
            [pd.read_parquet(f) for f in stats_files], ignore_index=True
        )
        # Solo columnas escalares (ignorar las de arrays: lidar/camera object counts)
        scalar_cols = ["key.segment_context_name", "key.frame_timestamp_micros"]
        rename_map = {
            "[StatsComponent].time_of_day": "time_of_day",
            "[StatsComponent].location": "location",
            "[StatsComponent].weather": "weather",
        }
        for raw_col, clean_col in rename_map.items():
            if raw_col in df_stats_raw.columns:
                scalar_cols.append(raw_col)

        df_stats = df_stats_raw[scalar_cols].rename(columns=rename_map)
        df_lidar = df_lidar.merge(
            df_stats,
            on=["key.segment_context_name", "key.frame_timestamp_micros"],
            how="left",
        )
        logger.info("Stats unidos: time_of_day, weather, location agregados.")
    else:
        logger.warning("No se encontraron parquets de stats en %s.", stats_dir)

    return df_lidar


# ---------------------------------------------------------------------------
# 2. extract_grayscale_images
# ---------------------------------------------------------------------------


def extract_grayscale_images(
    camera_image_dir: str,
    image_output_dir: str,
    stats_dir: str,
) -> pd.DataFrame:
    """Lee los parquets de camera_image, extrae los bytes JPEG de cada imagen,
    los convierte a escala de grises con OpenCV y los guarda como PNG.

    No se realiza ningun otro preprocesamiento (no se redimensiona, no se normaliza).
    Si el directorio esta vacio o no existe, devuelve un DataFrame vacio
    con el esquema correcto sin fallar el pipeline.

    Args:
        camera_image_dir: Directorio con los parquets de camera_image.
        image_output_dir: Directorio de salida para los PNG en gris.
        stats_dir:        Directorio con parquets de stats (para metadatos time_of_day/weather).

    Returns:
        DataFrame con image_path, context_name, timestamp_micros, camera_name,
        time_of_day, weather por cada imagen guardada.
    """
    import cv2  # importacion local para no fallar si cv2 no esta instalado en otros nodos

    cam_dir = Path(camera_image_dir)
    cam_files = sorted(cam_dir.glob("*.parquet")) if cam_dir.exists() else []

    if not cam_files:
        logger.warning(
            "No se encontraron parquets en %s; no se generan imagenes en escala de grises.",
            camera_image_dir,
        )
        return pd.DataFrame(columns=_IMAGE_METADATA_COLUMNS)

    # Cargar stats para metadata
    stats_files = sorted(Path(stats_dir).glob("*.parquet")) if Path(stats_dir).exists() else []
    if stats_files:
        df_stats = pd.concat(
            [pd.read_parquet(f) for f in stats_files], ignore_index=True
        ).rename(
            columns={
                "[StatsComponent].time_of_day": "time_of_day",
                "[StatsComponent].weather": "weather",
            }
        )
    else:
        df_stats = pd.DataFrame()

    output_root = Path(image_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    for parquet_file in cam_files:
        df_cam = pd.read_parquet(parquet_file)

        # Detectar columna de bytes de imagen automaticamente
        img_col = _find_image_column(df_cam)
        if img_col is None:
            logger.warning(
                "No se encontro columna de bytes de imagen en %s. Columnas disponibles: %s",
                parquet_file.name,
                df_cam.columns.tolist(),
            )
            continue

        logger.info(
            "Procesando %s (%d filas, columna imagen: '%s')",
            parquet_file.name,
            len(df_cam),
            img_col,
        )

        for _, row in df_cam.iterrows():
            img_bytes = row[img_col]
            if img_bytes is None:
                continue

            # Decodificar JPEG -> escala de grises
            try:
                img_arr = np.frombuffer(bytes(img_bytes), np.uint8)
                gray = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
            except Exception as exc:
                logger.warning("Error decodificando imagen: %s", exc)
                continue

            if gray is None:
                logger.warning(
                    "cv2.imdecode devolvio None para un frame de %s.", parquet_file.name
                )
                continue

            context_name = str(row.get("key.segment_context_name", parquet_file.stem))
            timestamp = row.get("key.frame_timestamp_micros", "")
            camera_name = str(row.get("key.camera_name", ""))

            filename = f"{context_name}_{timestamp}_{camera_name}.png"
            img_path = str(output_root / filename)
            cv2.imwrite(img_path, gray)

            # Obtener metadatos de stats
            time_of_day, weather = "", ""
            if not df_stats.empty:
                mask = (df_stats["key.segment_context_name"] == context_name) & (
                    df_stats["key.frame_timestamp_micros"] == timestamp
                )
                match = df_stats[mask]
                if not match.empty:
                    time_of_day = match.iloc[0].get("time_of_day", "")
                    weather = match.iloc[0].get("weather", "")

            records.append(
                {
                    "image_path": img_path,
                    "context_name": context_name,
                    "timestamp_micros": timestamp,
                    "camera_name": camera_name,
                    "time_of_day": time_of_day,
                    "weather": weather,
                }
            )

    logger.info("Imagenes en escala de grises guardadas: %d en %s", len(records), image_output_dir)
    return pd.DataFrame(records, columns=_IMAGE_METADATA_COLUMNS)


def _find_image_column(df: pd.DataFrame) -> str | None:
    """Detecta la columna que contiene bytes JPEG en un DataFrame de camera_image."""
    # Buscar primero por nombre canonico del Waymo v2
    for candidate in ["[CameraImageComponent].image", "image"]:
        if candidate in df.columns:
            return candidate
    # Fallback: primera columna de tipo object cuyo primer valor sea bytes
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(1)
            if len(sample) > 0 and isinstance(sample.iloc[0], (bytes, bytearray, memoryview)):
                return col
    return None


# ---------------------------------------------------------------------------
# 3. build_spark_feature_table
# ---------------------------------------------------------------------------


def build_spark_feature_table(
    waymo_features_raw: pd.DataFrame,
    spark_master: str,
) -> pd.DataFrame:
    """Aplica transformaciones representativas con PySpark sobre los features de Waymo.

    Transformaciones aplicadas (en memoria Spark, resultado exportado a pandas):
    - ``object_type_label``: tipo de objeto LiDAR (int → string legible).
    - ``box_volume``: volumen de la caja 3D (size.x * size.y * size.z).

    Si PySpark no esta instalado, registra un aviso y devuelve el DataFrame
    original sin modificacion para no bloquear el pipeline.

    Args:
        waymo_features_raw: Salida de ``load_and_merge_parquets``.
        spark_master:       URI del master de Spark (ej. ``"local[*]"``).

    Returns:
        DataFrame con las columnas originales mas ``object_type_label`` y ``box_volume``.
    """
    if waymo_features_raw.empty:
        logger.warning("waymo_features_raw esta vacio; build_spark_feature_table no aplica.")
        return waymo_features_raw

    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
    except ImportError:
        logger.warning(
            "PySpark no esta instalado. Ejecuta: python -m pip install pyspark>=3.5 "
            "Devolviendo los datos sin transformacion Spark."
        )
        # Aplica las mismas transformaciones en pandas como fallback
        return _spark_transforms_pandas_fallback(waymo_features_raw)

    # Seleccionar solo columnas escalares (Spark no acepta arrays/listas de numpy)
    scalar_df = _drop_array_columns(waymo_features_raw)

    spark = (
        SparkSession.builder
        .master(spark_master)
        .appName("waymo_v2")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        sdf = spark.createDataFrame(scalar_df)

        # Transformacion 1: tipo de objeto LiDAR int -> label string
        type_col = "[LiDARBoxComponent].type"
        if type_col in sdf.columns:
            sdf = sdf.withColumn(
                "object_type_label",
                F.when(F.col(f"`{type_col}`") == 1, "TYPE_VEHICLE")
                 .when(F.col(f"`{type_col}`") == 2, "TYPE_PEDESTRIAN")
                 .when(F.col(f"`{type_col}`") == 3, "TYPE_SIGN")
                 .when(F.col(f"`{type_col}`") == 4, "TYPE_CYCLIST")
                 .otherwise("TYPE_UNKNOWN"),
            )

        # Transformacion 2: volumen de la caja 3D
        sx = "[LiDARBoxComponent].box.size.x"
        sy = "[LiDARBoxComponent].box.size.y"
        sz = "[LiDARBoxComponent].box.size.z"
        if all(c in sdf.columns for c in [sx, sy, sz]):
            sdf = sdf.withColumn(
                "box_volume",
                F.col(f"`{sx}`") * F.col(f"`{sy}`") * F.col(f"`{sz}`"),
            )

        result = sdf.toPandas()
        logger.info(
            "Spark: transformaciones aplicadas. Filas: %d, columnas nuevas: object_type_label, box_volume.",
            len(result),
        )
    finally:
        spark.stop()

    return result


def _drop_array_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas cuyo primer valor no-nulo sea una lista o array de numpy."""
    drop = []
    for col in df.columns:
        sample = df[col].dropna().head(1)
        if len(sample) > 0 and isinstance(sample.iloc[0], (list, np.ndarray)):
            drop.append(col)
    if drop:
        logger.info("Columnas de arrays excluidas para Spark: %s", drop)
    return df.drop(columns=drop)


def _spark_transforms_pandas_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Replica las transformaciones Spark en pandas (fallback sin PySpark)."""
    result = df.copy()
    type_col = "[LiDARBoxComponent].type"
    if type_col in result.columns:
        result["object_type_label"] = result[type_col].map(_LIDAR_TYPE_MAP).fillna("TYPE_UNKNOWN")

    sx = "[LiDARBoxComponent].box.size.x"
    sy = "[LiDARBoxComponent].box.size.y"
    sz = "[LiDARBoxComponent].box.size.z"
    if all(c in result.columns for c in [sx, sy, sz]):
        result["box_volume"] = result[sx] * result[sy] * result[sz]

    logger.info("Fallback pandas: columnas object_type_label y box_volume calculadas sin Spark.")
    return result


# ---------------------------------------------------------------------------
# 4. build_waymo_v2_report
# ---------------------------------------------------------------------------


def build_waymo_v2_report(
    waymo_features: pd.DataFrame,
    image_metadata: pd.DataFrame,
) -> str:
    """Genera un reporte Markdown con las estadisticas clave del pipeline waymo_v2.

    Args:
        waymo_features: Salida de ``build_spark_feature_table``.
        image_metadata: Salida de ``extract_grayscale_images``.

    Returns:
        Texto Markdown del reporte.
    """
    lines = ["# Reporte del pipeline waymo_v2\n"]
    lines.append("Dataset: Waymo Open Dataset v2.0.1 (Parquet).\n")

    # --- Tabla de features ---
    lines.append("## Tabla de features (lidar_box + target + metadatos)\n")
    lines.append(f"- **Filas totales**: {len(waymo_features)}")
    lines.append(f"- **Columnas**: {len(waymo_features.columns)}")

    if "key.segment_context_name" in waymo_features.columns:
        n_segs = waymo_features["key.segment_context_name"].nunique()
        lines.append(f"- **Segmentos**: {n_segs}")

    if "tiene_camara" in waymo_features.columns and not waymo_features.empty:
        tc = waymo_features["tiene_camara"]
        lines.append(f"\n### Target `tiene_camara`")
        lines.append(f"| Clase | N | % |")
        lines.append(f"| --- | --- | --- |")
        lines.append(f"| Con camara (1) | {tc.sum()} | {tc.mean()*100:.1f}% |")
        lines.append(f"| Sin camara (0) | {(tc == 0).sum()} | {(tc == 0).mean()*100:.1f}% |")

    if "time_of_day" in waymo_features.columns and not waymo_features.empty:
        lines.append(f"\n### Distribucion por `time_of_day`")
        for val, cnt in waymo_features["time_of_day"].value_counts().items():
            lines.append(f"- **{val}**: {cnt} filas")

    if "weather" in waymo_features.columns and not waymo_features.empty:
        lines.append(f"\n### Distribucion por `weather`")
        for val, cnt in waymo_features["weather"].value_counts().items():
            lines.append(f"- **{val}**: {cnt} filas")

    if "object_type_label" in waymo_features.columns and not waymo_features.empty:
        lines.append(f"\n### Distribucion por `object_type_label` (Spark)")
        for val, cnt in waymo_features["object_type_label"].value_counts().items():
            lines.append(f"- **{val}**: {cnt} filas")

    if "box_volume" in waymo_features.columns and not waymo_features.empty:
        bv = waymo_features["box_volume"].dropna()
        lines.append(f"\n### Estadisticas de `box_volume` (m³, calculado en Spark)")
        lines.append(f"- Media: {bv.mean():.2f} | Mediana: {bv.median():.2f} | Max: {bv.max():.2f}")

    # --- Imagenes ---
    lines.append(f"\n## Imagenes en escala de grises\n")
    lines.append(f"- **Imagenes generadas**: {len(image_metadata)}")
    if not image_metadata.empty:
        if "time_of_day" in image_metadata.columns:
            lines.append(f"\n### Por `time_of_day`")
            for val, cnt in image_metadata["time_of_day"].value_counts().items():
                lines.append(f"- {val}: {cnt}")
        if "camera_name" in image_metadata.columns:
            lines.append(f"\n### Por camara")
            for val, cnt in image_metadata["camera_name"].value_counts().items():
                lines.append(f"- {val}: {cnt}")
    else:
        lines.append(
            "\n> No se generaron imagenes. Coloca los parquets de `camera_image` en "
            "`data/01_raw/waymo_v2/camera_image/` y vuelve a ejecutar el pipeline."
        )

    return "\n".join(lines)
