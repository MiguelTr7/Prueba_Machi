"""Tests de arranque del proyecto Kedro.

Verifican que el proyecto se puede inicializar y que los cuatro pipelines
quedan registrados y con nodos. Es una prueba de humo barata: si alguien
rompe un import o renombra el paquete, esto falla antes que cualquier
ejecucion real del pipeline.
"""
from pathlib import Path

import pytest
from kedro.framework.project import pipelines
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

PIPELINES_ESPERADOS = {"data_inventory", "data_processing", "ml", "retrasos_ml"}


@pytest.fixture(scope="module")
def proyecto():
    bootstrap_project(Path.cwd())
    return pipelines


class TestKedroRun:
    def test_se_registran_todos_los_pipelines(self, proyecto):
        assert PIPELINES_ESPERADOS <= set(proyecto)

    def test_el_pipeline_por_defecto_tiene_nodos(self, proyecto):
        assert len(proyecto["__default__"].nodes) > 0

    @pytest.mark.parametrize("nombre", sorted(PIPELINES_ESPERADOS))
    def test_cada_pipeline_tiene_nodos(self, proyecto, nombre):
        assert len(proyecto[nombre].nodes) > 0

    def test_la_sesion_se_crea_con_el_contexto_del_proyecto(self):
        bootstrap_project(Path.cwd())
        with KedroSession.create(project_path=Path.cwd()) as session:
            context = session.load_context()
            # Los parametros del PoC deben llegar desde conf/base.
            assert "retrasos_ml" in context.params
            assert context.params["retrasos_ml"]["aeropuerto_oaci"] == "SCTE"
