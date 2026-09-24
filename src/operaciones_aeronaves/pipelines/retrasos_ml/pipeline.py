"""Pipeline 'retrasos_ml' — Score de Riesgo Operativo por clima en SCTE.

Corre en paralelo al baseline (`data_processing` + `ml`), sin modificarlo:
parte de `vuelos_raw` y del clima diario, y termina en un modelo que entrega la
probabilidad de que un vuelo se retrase mas de 15 minutos.
"""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    construir_target_retraso,
    cruzar_vuelos_clima,
    evaluar_impacto_clima,
    filtrar_vuelos_scte,
    preparar_clima_scte,
    train_modelo_retrasos,
)
from .eda import analisis_exploratorio
from .segmentacion import perfilar_vuelos, segmentar_dias, segmentar_vuelos


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=filtrar_vuelos_scte,
                inputs=["vuelos_raw", "params:retrasos_ml"],
                outputs="vuelos_scte",
                name="filtrar_vuelos_scte_node",
            ),
            node(
                func=construir_target_retraso,
                inputs=["vuelos_scte", "params:retrasos_ml"],
                outputs="vuelos_scte_target",
                name="construir_target_retraso_node",
            ),
            node(
                func=preparar_clima_scte,
                inputs=["clima_diario_raw", "params:retrasos_ml"],
                outputs="clima_scte_features",
                name="preparar_clima_scte_node",
            ),
            node(
                func=cruzar_vuelos_clima,
                inputs=["vuelos_scte_target", "clima_scte_features"],
                outputs="vuelos_clima",
                name="cruzar_vuelos_clima_node",
            ),
            node(
                func=train_modelo_retrasos,
                inputs=["vuelos_clima", "params:retrasos_ml"],
                outputs=["metricas_retrasos", "busqueda_hiperparametros", "modelo_retrasos"],
                name="train_modelo_retrasos_node",
            ),
            node(
                func=evaluar_impacto_clima,
                inputs=["vuelos_clima", "modelo_retrasos"],
                outputs="efecto_clima_retraso",
                name="evaluar_impacto_clima_node",
            ),
            # --- Aprendizaje no supervisado sobre el mismo problema ---------
            node(
                func=perfilar_vuelos,
                inputs=["vuelos_clima", "params:retrasos_ml"],
                outputs="perfil_vuelos",
                name="perfilar_vuelos_node",
            ),
            node(
                func=segmentar_vuelos,
                inputs=["perfil_vuelos", "params:retrasos_ml"],
                outputs=["segmentos_vuelos", "modelo_segmentacion"],
                name="segmentar_vuelos_node",
            ),
            node(
                func=analisis_exploratorio,
                inputs=["vuelos_con_operaciones", "clima_scte_features", "vuelos_clima"],
                outputs="eda_report_md",
                name="analisis_exploratorio_node",
            ),
            node(
                func=segmentar_dias,
                inputs=["vuelos_clima", "params:retrasos_ml"],
                outputs="segmentos_dias",
                name="segmentar_dias_node",
            ),
        ]
    )
