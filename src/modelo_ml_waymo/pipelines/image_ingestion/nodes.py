"""
This is a boilerplate pipeline 'image_ingestion'
generated using Kedro 1.3.1
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset

logger = logging.getLogger(__name__)

_METADATA_COLUMNS = ["image_path", "context_name", "timestamp_micros", "camera_name", "time_of_day", "weather"]


# --------------------------------------------------------------------------
# 1. list_perception_tfrecords
# --------------------------------------------------------------------------


def list_perception_tfrecords(raw_tfrecord_dir: str) -> pd.DataFrame:
    """Index every ``.tfrecord`` file under ``raw_tfrecord_dir``, read-only.

    Nothing is downloaded and no tfrecord is opened here: only filesystem
    metadata is inspected. A missing or empty directory is not an error --
    it logs a warning and returns an empty (but correctly-shaped) table so
    downstream nodes can run as a no-op instead of failing the pipeline.

    Args:
        raw_tfrecord_dir: Directory to scan (e.g. ``data/01_raw/perception_tfrecords``).

    Returns:
        One row per tfrecord file with its path (relative to
        ``raw_tfrecord_dir``) and size in megabytes.
    """
    root = Path(raw_tfrecord_dir)
    columns = ["ruta_relativa", "tamano_mb"]
    if not root.exists():
        logger.warning("El directorio %s no existe; no hay tfrecords para procesar.", raw_tfrecord_dir)
        return pd.DataFrame(columns=columns)

    files = sorted(p for p in root.rglob("*.tfrecord") if p.is_file())
    if not files:
        logger.warning("No se encontraron archivos .tfrecord en %s.", raw_tfrecord_dir)
        return pd.DataFrame(columns=columns)

    rows = [
        {
            "ruta_relativa": f.relative_to(root).as_posix(),
            "tamano_mb": round(f.stat().st_size / (1024 * 1024), 4),
        }
        for f in files
    ]
    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------------------------------
# 2. extract_and_grayscale
# --------------------------------------------------------------------------


def extract_and_grayscale(
    tfrecord_index: pd.DataFrame, raw_tfrecord_dir: str, output_dir: str
) -> list[dict[str, Any]]:
    """Decode every camera image in every listed tfrecord to grayscale PNG.

    For each tfrecord, each ``Frame`` record is parsed and every
    ``frame.images[i]`` is decoded with OpenCV and written as-is to
    ``output_dir`` -- grayscale conversion is the only transformation
    applied, no resizing/cropping/normalisation. This node writes files as
    a side effect (the filenames are only known once each frame is parsed,
    so they cannot be pre-declared as a Kedro catalog entry); the per-image
    metadata gathered along the way is returned for
    ``build_image_metadata_table`` to persist.

    Args:
        tfrecord_index: Output of ``list_perception_tfrecords``.
        raw_tfrecord_dir: Directory the index's paths are relative to.
        output_dir: Directory grayscale PNGs are written into (created if missing).

    Returns:
        One record per saved image with image_path, context_name,
        timestamp_micros, camera_name, time_of_day and weather (the last
        two exactly as read from ``frame.context.stats``, never imputed).
    """
    if tfrecord_index.empty:
        logger.warning("El indice de tfrecords esta vacio; extract_and_grayscale no genera imagenes.")
        return []

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for relative_path in tfrecord_index["ruta_relativa"]:
        tfrecord_path = str(Path(raw_tfrecord_dir) / relative_path)
        dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="")
        for raw_record in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(raw_record.numpy()))
            records.extend(_extract_frame_images(frame, output_root))

    return records


def _extract_frame_images(frame: open_dataset.Frame, output_root: Path) -> list[dict[str, Any]]:
    """Decode and save every camera image in one ``Frame``; return their metadata rows."""
    context_name = frame.context.name
    timestamp_micros = frame.timestamp_micros
    time_of_day = frame.context.stats.time_of_day
    weather = frame.context.stats.weather

    rows: list[dict[str, Any]] = []
    for image in frame.images:
        camera_name = open_dataset.CameraName.Name.Name(image.name)
        img = cv2.imdecode(np.frombuffer(image.image, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(
                "No se pudo decodificar la imagen de %s/%s/%s; se omite.",
                context_name, timestamp_micros, camera_name,
            )
            continue

        filename = f"{context_name}_{timestamp_micros}_{camera_name}.png"
        image_path = str(output_root / filename)
        cv2.imwrite(image_path, img)

        rows.append(
            {
                "image_path": image_path,
                "context_name": context_name,
                "timestamp_micros": timestamp_micros,
                "camera_name": camera_name,
                "time_of_day": time_of_day,
                "weather": weather,
            }
        )
    return rows


# --------------------------------------------------------------------------
# 3. build_image_metadata_table
# --------------------------------------------------------------------------


def build_image_metadata_table(image_records: list[dict[str, Any]]) -> pd.DataFrame:
    """Consolidate the per-image records collected during extraction into one table.

    Args:
        image_records: Output of ``extract_and_grayscale``.

    Returns:
        One row per saved image with columns image_path, context_name,
        timestamp_micros, camera_name, time_of_day, weather. Values are
        kept exactly as extracted -- no null is imputed here.
    """
    return pd.DataFrame(image_records, columns=_METADATA_COLUMNS)
