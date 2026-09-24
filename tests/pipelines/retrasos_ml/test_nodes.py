"""Tests de la logica delicada del pipeline 'retrasos_ml'.

El target de retraso no viene en los datos: se reconstruye. Toda la validez del
PoC descansa en que esa reconstruccion sea correcta, y sus dos trampas —
el cruce de medianoche y los vuelos con mas de un slot diario — no se detectan
mirando metricas agregadas. De ahi estos tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from operaciones_aeronaves.pipelines.retrasos_ml.nodes import (
    _envolver,
    _referencia_por_slots,
    construir_target_retraso,
    cruzar_vuelos_clima,
    preparar_clima_scte,
)
from operaciones_aeronaves.pipelines.retrasos_ml.segmentacion import (
    _elegir_k,
    perfilar_vuelos,
)

PARAMS = {
    "slot_gap_min": 120,
    "slot_min_obs": 2,
    "grupo_min_obs": 3,
    "umbral_retraso_min": 15,
    "rumbo_pista": 170,
}


def _minutos(*horas: str) -> pd.Series:
    """'07:15' -> 435. Azucar para que los tests se lean como un itinerario."""
    valores = [int(h[:2]) * 60 + int(h[3:]) for h in horas]
    return pd.Series(valores, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# _envolver
# ─────────────────────────────────────────────────────────────────────────────

def test_envolver_deja_intactas_las_diferencias_chicas():
    assert _envolver(np.array([0.0, 15.0, -20.0])).tolist() == [0.0, 15.0, -20.0]


def test_envolver_corrige_el_cruce_de_medianoche():
    # 00:10 contra una referencia de 23:50: son 20 minutos tarde, no 1420.
    assert _envolver(np.array([10.0 - 1430.0])).item() == 20.0
    # 23:50 contra una referencia de 00:10: 20 minutos antes, no 1420 tarde.
    assert _envolver(np.array([1430.0 - 10.0])).item() == -20.0


# ─────────────────────────────────────────────────────────────────────────────
# _referencia_por_slots
# ─────────────────────────────────────────────────────────────────────────────

def test_slot_unico_devuelve_la_mediana():
    m = _minutos("11:11", "11:32", "11:05", "11:18", "11:07")
    ref = _referencia_por_slots(m, gap=120, min_obs=2)
    assert ref.nunique() == 1
    assert ref.iloc[0] == pytest.approx(11 * 60 + 11, abs=1)


def test_separa_dos_slots_del_mismo_numero_de_vuelo():
    """LATAM 61 llega ~07:16 los sabados y ~19:34 lunes y miercoles.

    Promediar ambos daria ~13:25, una hora en la que ese vuelo nunca opera.
    """
    m = _minutos("07:15", "07:08", "07:11", "19:36", "19:26", "19:34")
    ref = _referencia_por_slots(m, gap=120, min_obs=2)

    assert ref.nunique() == 2
    manana, tarde = sorted(ref.unique())
    assert manana == pytest.approx(7 * 60 + 11, abs=5)
    assert tarde == pytest.approx(19 * 60 + 34, abs=5)
    # Cada operacion se compara contra su propio slot, no contra el promedio.
    assert _envolver(m.values - ref.values).max() < 15


def test_slot_que_cruza_medianoche_queda_junto():
    """Un vuelo de las ~23:50 que a veces despega 00:10 es UN slot, no dos."""
    m = _minutos("23:50", "23:58", "00:05", "00:10", "23:45")
    ref = _referencia_por_slots(m, gap=120, min_obs=2)

    assert ref.nunique() == 1
    # La referencia cae cerca de medianoche, no a media tarde.
    assert min(ref.iloc[0], 1440 - ref.iloc[0]) < 30


def test_un_retraso_aislado_no_se_convierte_en_su_propio_slot():
    """Si un vuelo tardio formara slot propio, su retraso se auto-anularia.

    `min_obs=2` obliga a que un horario "habitual" tenga al menos dos
    observaciones; el vuelo solitario de las 03:00 se mide contra el slot real.
    """
    m = _minutos("21:00", "21:05", "20:55", "21:10", "03:00")
    ref = _referencia_por_slots(m, gap=120, min_obs=2)

    assert ref.nunique() == 1
    desvio = _envolver(m.values - ref.values)
    assert desvio[-1] > 300  # el vuelo de las 03:00 sale marcado como tardio


def test_una_sola_observacion_no_rompe():
    ref = _referencia_por_slots(_minutos("09:00"), gap=120, min_obs=2)
    assert ref.tolist() == [540.0]


# ─────────────────────────────────────────────────────────────────────────────
# construir_target_retraso
# ─────────────────────────────────────────────────────────────────────────────

def _bitacora(horas: list[str], fecha_base: str = "2024-03-01") -> pd.DataFrame:
    """Un mismo vuelo, mismo dia de la semana, en semanas consecutivas.

    Arranca el 1 de marzo (viernes) para que hasta cinco semanas caigan dentro
    del mismo mes: la agrupacion es por mes + dia de la semana, asi que cruzar
    a abril partiria el grupo en dos.
    """
    fechas = [
        pd.Timestamp(f"{fecha_base} {h}", tz="America/Santiago") + pd.Timedelta(weeks=i)
        for i, h in enumerate(horas)
    ]
    return pd.DataFrame({
        "dt_operacion": fechas,
        "aerolinea_dgac": "LAN",
        "numero_vuelo": "61",
        "tipo_operacion": "A",
        "es_internacional": False,
    })


def test_target_marca_retraso_sobre_el_umbral():
    # Cuatro vuelos puntuales y uno 40 minutos tarde.
    df = _bitacora(["11:00", "11:05", "10:58", "11:02", "11:45"])
    salida = construir_target_retraso(df, PARAMS)

    assert len(salida) == 5
    assert salida["retraso"].tolist() == [0, 0, 0, 0, 1]
    assert salida.loc[salida.retraso == 1, "desvio_min"].iloc[0] > 15


def test_target_no_marca_un_adelanto_como_retraso():
    """El target es asimetrico a proposito: solo cuenta llegar tarde."""
    df = _bitacora(["11:00", "11:05", "10:58", "11:02", "10:15"])
    salida = construir_target_retraso(df, PARAMS)
    assert salida["retraso"].sum() == 0


def test_descarta_grupos_con_muy_pocas_observaciones():
    """Con 2 observaciones no hay horario habitual estimable: se excluyen."""
    df = _bitacora(["11:00", "11:40"])
    salida = construir_target_retraso(df, PARAMS)
    assert salida.empty


def test_fecha_cruce_es_dia_sin_hora_ni_zona_horaria():
    df = _bitacora(["11:00", "11:05", "10:58"])
    salida = construir_target_retraso(df, PARAMS)

    assert salida["fecha_cruce"].dt.tz is None
    assert (salida["fecha_cruce"] == salida["fecha_cruce"].dt.normalize()).all()


# ─────────────────────────────────────────────────────────────────────────────
# preparar_clima_scte
# ─────────────────────────────────────────────────────────────────────────────

def _clima() -> pd.DataFrame:
    return pd.DataFrame({
        "fecha": pd.to_datetime(["2024-03-01", "2024-03-05", "2024-03-06"]),
        "tavg": [10.0, 12.0, 11.0],
        "tmin": [5.0, 7.0, 6.0],
        "tmax": [15.0, 18.0, 16.0],
        "prcp": [0.0, np.nan, 20.0],
        "snow": [np.nan] * 3,
        "wdir": [170.0, 260.0, 350.0],
        "wspd": [20.0, 20.0, 20.0],
        "wpgt": [np.nan] * 3,
        "pres": [1010.0, 1015.0, 1020.0],
        "tsun": [np.nan] * 3,
    })


def test_descarta_columnas_totalmente_vacias():
    salida = preparar_clima_scte(_clima(), PARAMS)
    for col in ("snow", "wpgt", "tsun"):
        assert col not in salida.columns


def test_lluvia_sin_registro_se_trata_como_cero():
    salida = preparar_clima_scte(_clima(), PARAMS)
    assert salida["prcp"].isna().sum() == 0
    assert salida.loc[1, "prcp"] == 0.0


def test_viento_cruzado_respeta_el_rumbo_de_la_pista():
    salida = preparar_clima_scte(_clima(), PARAMS)

    # Viento alineado con la pista (170°): no hay componente cruzada.
    assert salida.loc[0, "viento_cruzado"] == pytest.approx(0.0, abs=0.01)
    # Viento a 90° de la pista (260°): toda la velocidad es cruzada.
    assert salida.loc[1, "viento_cruzado"] == pytest.approx(20.0, abs=0.01)
    # Viento casi opuesto (350°): tampoco cruza, aunque sople al reves.
    assert salida.loc[2, "viento_cruzado"] == pytest.approx(0.0, abs=0.01)


def test_dia_lluvioso_es_flag_binario():
    salida = preparar_clima_scte(_clima(), PARAMS)
    assert salida["dia_lluvioso"].tolist() == [0, 0, 1]


# ─────────────────────────────────────────────────────────────────────────────
# cruzar_vuelos_clima
# ─────────────────────────────────────────────────────────────────────────────

def test_el_cruce_no_pierde_ni_duplica_vuelos():
    vuelos = construir_target_retraso(_bitacora(["11:00", "11:05", "10:58"]), PARAMS)
    clima = preparar_clima_scte(_clima(), PARAMS)

    salida = cruzar_vuelos_clima(vuelos, clima)

    # LEFT JOIN: el numero de vuelos se conserva exactamente.
    assert len(salida) == len(vuelos)
    # Solo la primera fecha existe en el clima de prueba; las otras quedan nulas
    # sin eliminar el vuelo.
    assert salida["tavg"].notna().sum() == 1


# ─────────────────────────────────────────────────────────────────────────────
# Seleccion de k en el aprendizaje no supervisado
# ─────────────────────────────────────────────────────────────────────────────

def test_descarta_el_k_que_aisla_un_puñado_de_casos():
    """El silhouette premia aislar outliers; la regla de tamano lo impide.

    Tres nubes compactas y dos puntos lejanos: con k=4 o k=5 el algoritmo
    aislaria los outliers en clusters diminutos, con un silhouette excelente
    y ninguna utilidad para decidir.
    """
    rng = np.random.default_rng(0)
    nubes = np.vstack([
        rng.normal(loc, 0.25, size=(40, 2)) for loc in ([0, 0], [8, 0], [0, 8])
    ])
    X = np.vstack([nubes, [[14.0, 14.0], [15.0, 15.0]]])

    k, diagnostico = _elegir_k(X, k_min=2, k_max=6, min_pct=0.05, random_state=42)

    # Sin la regla de tamano ganaria k=4, que aisla los dos outliers.
    assert int(diagnostico.loc[diagnostico.silhouette.idxmax(), "k"]) == 4
    assert k == 3
    # El diagnostico deja por escrito que hubo k con mejor silhouette y por que
    # se descartaron: es la evidencia que respalda la eleccion.
    descartados = diagnostico[~diagnostico.admisible]
    assert not descartados.empty
    assert (descartados.cluster_mas_chico_pct < 0.05).all()


def test_el_diagnostico_marca_el_k_elegido():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(loc, 0.3, size=(30, 2)) for loc in ([0, 0], [6, 6])])

    k, diagnostico = _elegir_k(X, k_min=2, k_max=5, min_pct=0.1, random_state=42)

    assert diagnostico.elegido.sum() == 1
    assert int(diagnostico.loc[diagnostico.elegido, "k"].iloc[0]) == k


def test_perfil_de_vuelos_descarta_slots_con_pocas_operaciones():
    """Un slot con 3 vuelos no tiene comportamiento estable que describir."""
    frecuente = _bitacora(["11:00", "11:05", "10:58", "11:02", "11:45"])
    salida = construir_target_retraso(frecuente, PARAMS)
    salida["hora_programada"] = salida["hora_referencia_min"] / 60.0

    perfil = perfilar_vuelos(salida, {"min_ops_slot": 30})
    assert perfil.empty

    perfil = perfilar_vuelos(salida, {"min_ops_slot": 3})
    assert len(perfil) == 1
    assert perfil.iloc[0]["n_operaciones"] == 5
    # La variabilidad es el rango intercuartil del desvio, no su promedio.
    assert perfil.iloc[0]["variabilidad_min"] >= 0
