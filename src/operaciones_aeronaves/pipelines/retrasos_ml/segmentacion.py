"""
Aprendizaje NO supervisado aplicado al problema de retrasos en SCTE.

El clustering de aeropuertos del pipeline `ml` responde una pregunta distinta
(como se parecen entre si los aeropuertos de Chile). Aqui el objetivo es otro:
descubrir estructura *dentro del problema de negocio*, y se ataca por dos vias
complementarias que responden preguntas diferentes.

1. SEGMENTACION DE VUELOS (`segmentar_vuelos`)
   Agrupa los slots de vuelo por su comportamiento operativo historico.
   Pregunta: *¿que vuelos concentran el riesgo?*
   Es segmentacion **descriptiva**: incluye la puntualidad entre las variables
   de agrupacion a proposito, porque el objetivo es describir y priorizar, no
   predecir. Decir "los clusters difieren en puntualidad" seria circular; lo
   que si aporta valor es **cuales** vuelos caen en el grupo critico y que tan
   concentrado esta el riesgo.

2. SEGMENTACION DE DIAS (`segmentar_dias`)
   Agrupa los dias por sus condiciones — carga operativa y clima — **sin mirar
   el resultado**. Pregunta: *¿existe un arquetipo de dia malo reconocible por
   sus condiciones?*
   Aqui la puntualidad se usa solo para interpretar los grupos ya formados, asi
   que el resultado si es una prueba legitima: es una validacion independiente
   del hallazgo del modelo supervisado, por una via que no usa etiquetas.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

_IMAGES_DIR = Path("images")

_COLORES = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def _elegir_k(
    X: np.ndarray, k_min: int, k_max: int, min_pct: float, random_state: int
) -> tuple[int, pd.DataFrame]:
    """Elige k por silhouette, descartando particiones degeneradas.

    El silhouette por si solo suele premiar un k alto que aisla un puñado de
    casos raros en un cluster propio: matematicamente compacto, inutil para
    decidir. La regla anade una condicion de tamano — **ningun cluster puede
    quedar por debajo de `min_pct` del total** — y entre los k que la cumplen
    se queda con el de mejor silhouette.
    """
    filas = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        etiquetas = km.fit_predict(X)
        _, conteos = np.unique(etiquetas, return_counts=True)
        pct_minimo = conteos.min() / len(X)
        filas.append({
            "k": k,
            "inercia": round(float(km.inertia_), 2),
            "silhouette": round(float(silhouette_score(X, etiquetas)), 4),
            "cluster_mas_chico_pct": round(float(pct_minimo), 4),
            "admisible": bool(pct_minimo >= min_pct),
        })

    diagnostico = pd.DataFrame(filas)
    admisibles = diagnostico[diagnostico.admisible]
    if admisibles.empty:
        logger.warning("Ningun k cumple el tamano minimo de cluster; se usa el mejor silhouette")
        admisibles = diagnostico

    k_elegido = int(admisibles.loc[admisibles.silhouette.idxmax(), "k"])
    diagnostico["elegido"] = diagnostico.k == k_elegido
    logger.info("k elegido = %d\n%s", k_elegido, diagnostico.to_string(index=False))
    return k_elegido, diagnostico


def _grafico_seleccion_k(diagnostico: pd.DataFrame, k: int, titulo: str, ruta: Path) -> None:
    """Curva del codo y silhouette, marcando el k elegido y los descartados."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(diagnostico.k, diagnostico.inercia, marker="o", linewidth=2, color=_COLORES[0])
    ax1.axvline(k, color=_COLORES[1], linestyle="--", label=f"k={k} elegido")
    ax1.set_xlabel("Numero de clusters (k)")
    ax1.set_ylabel("Inercia (WCSS)")
    ax1.set_title("Curva del codo")
    ax1.legend()

    colores = [
        _COLORES[1] if fila.k == k else (_COLORES[0] if fila.admisible else "#BBBBBB")
        for fila in diagnostico.itertuples()
    ]
    ax2.bar(diagnostico.k, diagnostico.silhouette, color=colores, alpha=0.9)
    ax2.set_xlabel("Numero de clusters (k)")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title("Silhouette\n(gris = descartado por tener un cluster demasiado chico)")

    fig.suptitle(titulo, fontsize=13)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    logger.info("Guardado %s", ruta)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SEGMENTACION DE VUELOS — ¿que vuelos concentran el riesgo?
# ─────────────────────────────────────────────────────────────────────────────

_FEATURES_VUELO = ["hora_programada", "tasa_retraso", "variabilidad_min", "pct_severo", "n_operaciones"]


def perfilar_vuelos(vuelos_clima: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Resume el comportamiento historico de cada slot de vuelo.

    Un "slot" es la combinacion aerolinea + numero de vuelo + tipo de operacion:
    la unidad que un jefe de operaciones puede efectivamente intervenir.
    """
    agrupado = vuelos_clima.groupby(["aerolinea_dgac", "numero_vuelo", "tipo_operacion"])

    perfil = agrupado.agg(
        n_operaciones=("retraso", "size"),
        hora_programada=("hora_programada", "mean"),
        tasa_retraso=("retraso", "mean"),
        desvio_mediano=("desvio_min", "median"),
        # Rango intercuartil: cuanto se mueve este vuelo respecto de si mismo.
        # Un vuelo puede ser tardio pero predecible, o puntual pero erratico.
        variabilidad_min=("desvio_min", lambda s: s.quantile(0.75) - s.quantile(0.25)),
        pct_severo=("desvio_min", lambda s: (s > 60).mean()),
        pct_internacional=("es_internacional", "mean"),
    ).reset_index()

    minimo = params["min_ops_slot"]
    antes = len(perfil)
    perfil = perfil[perfil.n_operaciones >= minimo].reset_index(drop=True)
    logger.info(
        "Perfil de vuelos: %d slots con >= %d operaciones (de %d); cubren %d vuelos",
        len(perfil), minimo, antes, perfil.n_operaciones.sum(),
    )
    return perfil


def segmentar_vuelos(perfil_vuelos: pd.DataFrame, params: dict) -> tuple:
    """K-Means sobre el comportamiento operativo de cada slot de vuelo.

    Genera `rt_05_seleccion_k_vuelos.png` y `rt_06_segmentos_vuelos.png`.
    Devuelve (perfil con su segmento, artefacto con el modelo y el diagnostico).
    """
    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    X = perfil_vuelos[_FEATURES_VUELO].copy()
    # El volumen es muy asimetrico (30 a 1564 operaciones): en logaritmo deja
    # de dominar la distancia euclidiana.
    X["n_operaciones"] = np.log1p(X["n_operaciones"])

    escalador = StandardScaler()
    Xs = escalador.fit_transform(X)

    k, diagnostico = _elegir_k(
        Xs, params["k_min"], params["k_max"], params["min_pct_cluster"], params["random_state"]
    )
    _grafico_seleccion_k(
        diagnostico, k,
        "Cuantos perfiles de vuelo distintos hay en El Tepual",
        _IMAGES_DIR / "rt_05_seleccion_k_vuelos.png",
    )

    modelo = KMeans(n_clusters=k, random_state=params["random_state"], n_init=20)
    etiquetas = modelo.fit_predict(Xs)

    segmentos = perfil_vuelos.copy()
    segmentos["segmento"] = etiquetas

    pca = PCA(n_components=2, random_state=params["random_state"])
    coords = pca.fit_transform(Xs)
    segmentos["pca_1"], segmentos["pca_2"] = coords[:, 0], coords[:, 1]

    # El segmento critico se nombra por su tasa de retraso, no por su indice:
    # el numero que asigna K-Means no significa nada entre ejecuciones.
    resumen = (
        segmentos.groupby("segmento")
        .agg(
            n_slots=("n_operaciones", "size"),
            vuelos_cubiertos=("n_operaciones", "sum"),
            hora_media=("hora_programada", "mean"),
            tasa_retraso=("tasa_retraso", "mean"),
            variabilidad_min=("variabilidad_min", "mean"),
            pct_severo=("pct_severo", "mean"),
        )
        .sort_values("tasa_retraso", ascending=False)
        .reset_index()
    )
    critico = int(resumen.iloc[0]["segmento"])
    segmentos["es_segmento_critico"] = (segmentos.segmento == critico).astype("int8")

    logger.info("Segmentos de vuelo:\n%s", resumen.round(3).to_string(index=False))
    total_vuelos = segmentos.n_operaciones.sum()
    vuelos_criticos = resumen.iloc[0]["vuelos_cubiertos"]
    logger.info(
        "Segmento critico: %d slots (%.1f%% de los slots) que concentran %.1f%% de los vuelos "
        "con una tasa de retraso de %.1f%% (base %.1f%%)",
        resumen.iloc[0]["n_slots"], 100 * resumen.iloc[0]["n_slots"] / len(segmentos),
        100 * vuelos_criticos / total_vuelos, 100 * resumen.iloc[0]["tasa_retraso"],
        100 * segmentos.tasa_retraso.mean(),
    )

    _grafico_segmentos_vuelos(segmentos, resumen, pca, critico)

    artefacto = {
        "modelo": modelo,
        "escalador": escalador,
        "pca": pca,
        "features": _FEATURES_VUELO,
        "k": k,
        "diagnostico_k": diagnostico,
        "resumen": resumen,
        "segmento_critico": critico,
    }
    return segmentos, artefacto


def _grafico_segmentos_vuelos(
    segmentos: pd.DataFrame, resumen: pd.DataFrame, pca: PCA, critico: int
) -> None:
    """Mapa PCA de los segmentos y su perfil de riesgo."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i, seg in enumerate(sorted(segmentos.segmento.unique())):
        mascara = segmentos.segmento == seg
        etiqueta = f"Segmento {seg}" + (" (critico)" if seg == critico else "")
        ax1.scatter(
            segmentos.loc[mascara, "pca_1"], segmentos.loc[mascara, "pca_2"],
            color=_COLORES[i % len(_COLORES)], s=np.sqrt(segmentos.loc[mascara, "n_operaciones"]) * 3,
            alpha=0.75, edgecolors="grey", linewidth=0.3, label=etiqueta,
        )
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% de la varianza)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax1.set_title("Segmentos de vuelo en el plano principal\n(tamano ~ volumen de operaciones)")
    ax1.legend(fontsize=8)

    orden = resumen.sort_values("tasa_retraso")
    colores = [_COLORES[int(s) % len(_COLORES)] for s in orden.segmento]
    barras = ax2.barh(
        [f"Seg. {int(s)}\n({int(n)} slots)" for s, n in zip(orden.segmento, orden.n_slots)],
        orden.tasa_retraso * 100, color=colores, alpha=0.9,
    )
    base = segmentos.tasa_retraso.mean() * 100
    ax2.axvline(base, color="grey", linestyle="--", label=f"Media global {base:.1f}%")
    for barra, variabilidad in zip(barras, orden.variabilidad_min):
        ax2.text(
            barra.get_width() + 0.4, barra.get_y() + barra.get_height() / 2,
            f"±{variabilidad:.0f} min", va="center", fontsize=8,
        )
    ax2.set_xlabel("% de vuelos retrasados")
    ax2.set_title("Riesgo por segmento\n(la etiqueta muestra la variabilidad tipica)")
    ax2.legend(fontsize=8)

    fig.suptitle("¿Que vuelos concentran el riesgo en El Tepual?", fontsize=13)
    fig.tight_layout()
    ruta = _IMAGES_DIR / "rt_06_segmentos_vuelos.png"
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    logger.info("Guardado %s", ruta)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SEGMENTACION DE DIAS — ¿existe un arquetipo de dia malo?
# ─────────────────────────────────────────────────────────────────────────────

_FEATURES_DIA = ["n_operaciones", "n_aerolineas", "prcp", "viento_cruzado",
                 "wspd", "tavg", "amplitud_termica", "pres"]


def segmentar_dias(vuelos_clima: pd.DataFrame, params: dict) -> pd.DataFrame:
    """K-Means sobre las condiciones de cada dia, sin mirar la puntualidad.

    Esta es la prueba limpia: si las condiciones de operacion — carga y clima —
    contuvieran informacion sobre el retraso, los grupos formados solo con
    ellas deberian separarse tambien en puntualidad. La tasa de retraso entra
    despues, unicamente para etiquetar los grupos ya formados.

    Genera `rt_07_segmentos_dias.png`. Devuelve el perfil diario con su grupo.
    """
    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    dias = (
        vuelos_clima.groupby("fecha_cruce")
        .agg(
            n_operaciones=("retraso", "size"),
            n_aerolineas=("aerolinea_dgac", "nunique"),
            prcp=("prcp", "first"),
            viento_cruzado=("viento_cruzado", "first"),
            wspd=("wspd", "first"),
            tavg=("tavg", "first"),
            amplitud_termica=("amplitud_termica", "first"),
            pres=("pres", "first"),
            # Solo para interpretar: NO entra al clustering.
            tasa_retraso=("retraso", "mean"),
        )
        .dropna(subset=_FEATURES_DIA)
    )
    dias = dias[dias.n_operaciones >= params["min_ops_dia"]].reset_index()

    X = dias[_FEATURES_DIA].copy()
    X["prcp"] = np.log1p(X["prcp"])  # la lluvia diaria es muy asimetrica
    Xs = StandardScaler().fit_transform(X)

    k, diagnostico = _elegir_k(
        Xs, params["k_min"], params["k_max"], params["min_pct_cluster"], params["random_state"]
    )

    # La conclusion no puede depender de haber elegido un k afortunado: se mide
    # la brecha de puntualidad entre el mejor y el peor grupo para CADA k.
    brechas = []
    for k_i in range(params["k_min"], params["k_max"] + 1):
        etiquetas = KMeans(
            n_clusters=k_i, random_state=params["random_state"], n_init=20
        ).fit_predict(Xs)
        tasas = dias.groupby(etiquetas).tasa_retraso.mean()
        brechas.append({"k": k_i, "brecha_puntos": round(100 * (tasas.max() - tasas.min()), 2)})
    brechas = pd.DataFrame(brechas)
    logger.info(
        "Brecha de puntualidad entre el mejor y el peor tipo de dia, por k:\n%s",
        brechas.to_string(index=False),
    )
    logger.info(
        "La brecha nunca supera %.1f puntos: agrupar los dias por sus condiciones "
        "no separa los dias buenos de los malos, sea cual sea el numero de grupos.",
        brechas.brecha_puntos.max(),
    )

    modelo = KMeans(n_clusters=k, random_state=params["random_state"], n_init=20)
    dias["tipo_dia"] = modelo.fit_predict(Xs)

    resumen = (
        dias.groupby("tipo_dia")
        .agg(
            n_dias=("tasa_retraso", "size"),
            operaciones=("n_operaciones", "mean"),
            lluvia_mm=("prcp", "mean"),
            viento_cruzado=("viento_cruzado", "mean"),
            temperatura=("tavg", "mean"),
            tasa_retraso=("tasa_retraso", "mean"),
        )
        .reset_index()
    )
    dispersion = resumen.tasa_retraso.max() - resumen.tasa_retraso.min()
    logger.info("Tipos de dia (agrupados solo por condiciones):\n%s",
                resumen.round(3).to_string(index=False))
    logger.info(
        "Diferencia de puntualidad entre el mejor y el peor tipo de dia: %.1f puntos. "
        "Si las condiciones explicaran el retraso, esta brecha seria grande.",
        100 * dispersion,
    )

    _grafico_segmentos_dias(dias, resumen, brechas)
    return dias


def _grafico_segmentos_dias(
    dias: pd.DataFrame, resumen: pd.DataFrame, brechas: pd.DataFrame
) -> None:
    """Perfil de cada tipo de dia, su puntualidad, y la robustez del resultado."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

    ax1.scatter(
        dias.viento_cruzado, np.log1p(dias.prcp),
        c=[_COLORES[int(t) % len(_COLORES)] for t in dias.tipo_dia],
        s=dias.n_operaciones * 0.8, alpha=0.6, edgecolors="grey", linewidth=0.2,
    )
    ax1.set_xlabel("Viento cruzado (km/h)")
    ax1.set_ylabel("log(1 + lluvia diaria en mm)")
    ax1.set_title("Los dias se agrupan por sus condiciones\n(tamano ~ operaciones del dia)")

    base = dias.tasa_retraso.mean() * 100
    colores = [_COLORES[int(t) % len(_COLORES)] for t in resumen.tipo_dia]
    ax2.bar(
        [f"Tipo {int(t)}\n({int(n)} dias)" for t, n in zip(resumen.tipo_dia, resumen.n_dias)],
        resumen.tasa_retraso * 100, color=colores, alpha=0.9,
    )
    ax2.axhline(base, color="#DD8452", linestyle="--", label=f"Media global {base:.1f}%")
    ax2.set_ylabel("% de vuelos retrasados")
    ax2.set_ylim(0, max(resumen.tasa_retraso) * 100 * 1.4)
    ax2.set_title("...pero eso casi no cambia la puntualidad")
    ax2.legend(fontsize=8)

    # Tercer panel: la conclusion no depende de haber elegido un k afortunado.
    ax3.plot(brechas.k, brechas.brecha_puntos, marker="o", linewidth=2, color=_COLORES[0])
    ax3.fill_between(brechas.k, brechas.brecha_puntos, alpha=0.15, color=_COLORES[0])
    ax3.axhline(10, color=_COLORES[3], linestyle="--",
                label="10 pts: brecha que haria util\nagrupar los dias")
    ax3.set_xlabel("Numero de grupos (k)")
    ax3.set_ylabel("Brecha mejor vs peor tipo (puntos)")
    ax3.set_ylim(0, 12)
    ax3.set_title("El resultado no depende del k elegido")
    ax3.legend(fontsize=8)

    fig.suptitle("¿Existe un arquetipo de dia malo reconocible por sus condiciones?", fontsize=13)
    fig.tight_layout()
    ruta = _IMAGES_DIR / "rt_07_segmentos_dias.png"
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    logger.info("Guardado %s", ruta)
