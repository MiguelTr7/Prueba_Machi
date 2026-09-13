"""Pipeline 'ml' — EDA avanzado, clustering no supervisado y clasificacion supervisada."""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    cluster_aeropuertos,
    eda_avanzado,
    evaluar_clasificador,
    evaluar_clustering,
    perfil_aeropuertos,
    train_clasificador,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=perfil_aeropuertos,
                inputs="vuelos_con_operaciones",
                outputs="aeropuertos_perfil",
                name="perfil_aeropuertos_node",
            ),
            node(
                func=eda_avanzado,
                inputs=["vuelos_con_operaciones", "aeropuertos_perfil"],
                outputs="eda_avanzado_manifest",
                name="eda_avanzado_node",
            ),
            node(
                func=cluster_aeropuertos,
                inputs="aeropuertos_perfil",
                outputs=["cluster_labels", "modelo_kmeans"],
                name="cluster_aeropuertos_node",
            ),
            node(
                func=evaluar_clustering,
                inputs="aeropuertos_perfil",
                outputs="cluster_metricas",
                name="evaluar_clustering_node",
            ),
            node(
                func=train_clasificador,
                inputs="vuelos_con_operaciones",
                outputs=["metricas_supervisado", "modelo_rf"],
                name="train_clasificador_node",
            ),
            node(
                func=evaluar_clasificador,
                inputs="vuelos_con_operaciones",
                outputs="metricas_cv_supervisado",
                name="evaluar_clasificador_node",
            ),
        ]
    )
