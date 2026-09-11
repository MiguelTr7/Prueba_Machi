"""Pipeline waymo_v2 — procesamiento del Waymo Open Dataset v2.0.1 en Parquet."""
from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
