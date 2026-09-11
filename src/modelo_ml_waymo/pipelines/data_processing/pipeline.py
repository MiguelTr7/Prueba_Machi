"""Pipeline 'data_processing' — ingesta, preprocesamiento y EDA de datos JAC."""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_eda_plots, join_con_aeropuertos, preprocess_vuelos


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_vuelos,
                inputs="vuelos_raw",
                outputs="vuelos_preprocessed",
                name="preprocess_vuelos_node",
            ),
            node(
                func=join_con_aeropuertos,
                inputs=["vuelos_preprocessed", "aeropuertos_raw"],
                outputs="vuelos_con_operaciones",
                name="join_con_aeropuertos_node",
            ),
            node(
                func=generate_eda_plots,
                inputs="vuelos_con_operaciones",
                outputs="eda_figura_manifest",
                name="generate_eda_plots_node",
            ),
        ]
    )
