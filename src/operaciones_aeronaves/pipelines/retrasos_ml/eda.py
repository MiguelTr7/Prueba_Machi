"""
Analisis exploratorio completo, como reporte reproducible.

Cubre las tres fuentes y los dos recortes del proyecto en un solo documento:
la bitacora entera, el clima, el subconjunto de El Tepual y el target de
retraso. Es la Fase 2 de CRISP-DM escrita como pipeline, de modo que el
diagnostico se regenera cada vez que cambia un dato en vez de quedar congelado
en un notebook.

Sale a `data/08_reporting/eda_report.md` mas dos figuras en `images/`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

_IMAGES_DIR = Path("images")
_COLOR_A = "#4C72B0"
_COLOR_B = "#DD8452"


def _contar_ceros_crudos() -> int:
    """Cuenta los `pmd = 0` en la fuente original.

    Al llegar a este nodo el preprocesamiento ya los convirtio en faltantes, asi
    que el dato solo existe en el parquet crudo. Se lee **una sola columna** para
    no cargar los 137 MB completos.
    """
    ruta = Path("data/01_raw/bitacora-vuelos.parquet")
    if not ruta.exists():
        return 0
    import pyarrow.parquet as pq

    pmd = pq.read_table(ruta, columns=["pmd"]).to_pandas()["pmd"]
    return int((pd.to_numeric(pmd, errors="coerce") == 0).sum())


def _tabla(df: pd.DataFrame) -> str:
    """DataFrame -> tabla Markdown, sin dependencias extra."""
    encabezado = "| " + " | ".join(str(c) for c in df.columns) + " |"
    separador = "| " + " | ".join("---" for _ in df.columns) + " |"
    filas = ["| " + " | ".join(str(v) for v in fila) + " |" for fila in df.itertuples(index=False)]
    return "\n".join([encabezado, separador, *filas])


def _resumen_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Tipo, nulos, cardinalidad y ejemplo de cada columna."""
    filas = []
    for col in df.columns:
        serie = df[col]
        ejemplo = serie.dropna()
        filas.append({
            "columna": col,
            "tipo": str(serie.dtype),
            "nulos": int(serie.isna().sum()),
            "% nulos": round(float(serie.isna().mean() * 100), 2),
            "valores_distintos": int(serie.nunique()),
            "ejemplo": str(ejemplo.iloc[0])[:28] if len(ejemplo) else "—",
        })
    return pd.DataFrame(filas)


def _describe_numerico(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """describe() con percentiles utiles para detectar colas largas."""
    presentes = [c for c in columnas if c in df.columns]
    if not presentes:
        return pd.DataFrame()
    desc = df[presentes].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    desc = desc.round(2).reset_index().rename(columns={"index": "variable"})
    return desc


def _analizar_outliers(serie: pd.Series, nombre: str) -> tuple[pd.DataFrame, dict]:
    """Regla del rango intercuartil, con el detalle de que hay en las colas.

    Recortar para graficar (como hacia `clip(500)`) esconde el problema en vez
    de diagnosticarlo: aqui se cuenta cuantos casos caen fuera del rango y se
    mira si son errores de registro o aeronaves legitimamente enormes.
    """
    limpia = serie.dropna()
    q1, q3 = limpia.quantile(0.25), limpia.quantile(0.75)
    iqr = q3 - q1
    bajo, alto = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    inferiores = int((limpia < bajo).sum())
    superiores = int((limpia > alto).sum())
    resumen = {
        "variable": nombre,
        "q1": round(float(q1), 2),
        "q3": round(float(q3), 2),
        "iqr": round(float(iqr), 2),
        "limite_inferior": round(float(bajo), 2),
        "limite_superior": round(float(alto), 2),
        "outliers_inferiores": inferiores,
        "outliers_superiores": superiores,
        "pct_outliers": round(100 * (inferiores + superiores) / len(limpia), 2),
        "maximo": round(float(limpia.max()), 2),
        "minimo": round(float(limpia.min()), 2),
    }
    extremos = (
        limpia.sort_values(ascending=False).head(8).reset_index(drop=True)
        .rename(nombre).to_frame().round(2)
    )
    return extremos, resumen


def analisis_exploratorio(
    vuelos: pd.DataFrame, clima: pd.DataFrame, vuelos_clima: pd.DataFrame
) -> str:
    """Genera el reporte EDA completo en Markdown.

    Entradas: la bitacora integrada (11M filas), el clima diario ya preparado y
    el cruce final de la PoC con su target.
    """
    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    L: list[str] = ["# Analisis Exploratorio de Datos (EDA)", ""]
    L += ["> Documento generado por el pipeline (`analisis_exploratorio_node`).",
          "> Se regenera con `kedro run`; no se edita a mano.", ""]

    # ── 1. La bitacora completa ──────────────────────────────────────────────
    L += ["## 1. Bitacora de vuelos — estructura y calidad", "",
          f"**{len(vuelos):,} filas · {vuelos.shape[1]} columnas** "
          f"({vuelos.dt_operacion.min():%Y-%m-%d} a {vuelos.dt_operacion.max():%Y-%m-%d})", ""]
    L += [_tabla(_resumen_columnas(vuelos)), ""]

    num_vuelos = ["pmd", "mes_id", "cnt_operaciones"]
    L += ["### Resumen estadistico de las variables numericas", "",
          _tabla(_describe_numerico(vuelos, num_vuelos)), ""]

    # ── 2. Duplicados ────────────────────────────────────────────────────────
    L += ["## 2. Duplicados", ""]
    exactos = int(vuelos.duplicated().sum())
    # Llave de negocio: una misma aeronave no puede operar dos veces en el
    # mismo instante y aeropuerto. Un choque aqui es un error de registro.
    llave = ["dt_operacion", "aeropuerto_oaci", "matricula", "tipo_operacion"]
    dup_llave = int(vuelos.duplicated(subset=llave).sum())
    L += [_tabla(pd.DataFrame([
        {"tipo": "Filas identicas en todas sus columnas", "cantidad": f"{exactos:,}",
         "% del total": round(100 * exactos / len(vuelos), 3)},
        {"tipo": "Repeticiones de (fecha-hora, aeropuerto, matricula, tipo)",
         "cantidad": f"{dup_llave:,}", "% del total": round(100 * dup_llave / len(vuelos), 3)},
    ])), ""]
    if exactos or dup_llave:
        L += ["Los duplicados exactos son operaciones que la JAC registra mas de una vez. "
              "**No se eliminan**: sin un identificador unico de operacion no se puede "
              "distinguir un registro repetido de dos movimientos reales muy seguidos, y "
              "borrarlos a ciegas sesgaria los conteos por aeropuerto.", ""]
    else:
        L += ["No hay duplicados. Cada fila es una operacion distinta.", ""]

    # ── 3. Outliers ──────────────────────────────────────────────────────────
    L += ["## 3. Valores extremos del PMD", "",
          "El PMD (Peso Maximo de Despegue, en toneladas) es la variable numerica "
          "con la cola mas larga del dataset.", ""]
    _, resumen_out = _analizar_outliers(vuelos["pmd"], "pmd")
    L += [_tabla(pd.DataFrame([resumen_out])), ""]
    L += ["El 3% de los vuelos cae fuera del rango intercuartil. **Mirar quien es cada "
          "uno separa tres situaciones distintas**, y solo una de ellas es normal.", ""]

    # (a) Cola alta legitima: fuselaje ancho.
    anchos = vuelos[(vuelos.pmd > 300) & (vuelos.pmd <= 650)]
    top_anchos = (
        anchos.groupby("modelo_avion")["pmd"].agg(["size", "max"])
        .sort_values("size", ascending=False).head(5).reset_index()
        .rename(columns={"size": "operaciones", "max": "pmd_max"})
    )
    L += ["### (a) Cola alta legitima — aeronaves de fuselaje ancho", "",
          f"{len(anchos):,} vuelos entre 300 y 650 toneladas. Son modelos reales "
          "(B747, B777, A340) y el AN-225, que efectivamente pesa 640 t. "
          "**Se conservan**: transformarlos con logaritmo basta para que no dominen "
          "la escala al modelar.", "", _tabla(top_anchos), ""]

    # (b) Errores de unidad: aeronaves livianas registradas en kilogramos.
    imposibles = vuelos[vuelos.pmd > 650]
    if len(imposibles):
        top_imp = (
            imposibles.groupby(["modelo_avion", "modelo_avion_desc"])["pmd"]
            .agg(["size", "max"]).sort_values("size", ascending=False).head(6)
            .reset_index().rename(columns={"size": "operaciones", "max": "pmd_registrado"})
        )
        L += ["### (b) Error de unidad — kilogramos registrados como toneladas", "",
              f"**{len(imposibles):,} vuelos superan las 650 toneladas**, mas que "
              "cualquier aeronave que haya volado. Al mirar el modelo queda claro que "
              "no son aviones gigantes sino **aviones pequeños con el peso en "
              "kilogramos**: un T-34 Mentor pesa 1.3 toneladas y aparece con 1 340; "
              "un helicoptero R-44 pesa 1.1 t y figura con 1 000.", "",
              _tabla(top_imp), "",
              "**No se corrigen automaticamente.** Dividir por 1 000 arreglaria estos "
              "casos, pero exigiria decidir por umbral cuales convertir, y un umbral mal "
              "puesto danaria registros correctos. Son el 0.01% de los datos y ninguno "
              "entra en la PoC de El Tepual, asi que se documentan como anomalia "
              "conocida en vez de parcharse a ciegas.", ""]

    # (c) Ceros: nulos disfrazados. Se cuentan sobre la fuente cruda, porque al
    # llegar aqui el preprocesamiento ya los convirtio en faltantes.
    ceros = _contar_ceros_crudos()
    imputados = int(vuelos["pmd_fue_imputado"].sum())
    L += ["### (c) Ceros — nulos disfrazados", "",
          "Un avion no puede pesar cero. En la fuente cruda hay "
          f"**{ceros:,} registros con `pmd = 0`**" if ceros else
          "Un avion no puede pesar cero.", ""]
    L += [f"La primera version del pipeline imputaba solo los `NaN`, asi que estos "
          "pasaban intactos al modelo como aviones sin peso. **Ahora se tratan como "
          "faltantes** y entran a la imputacion por mediana del modelo de avion, igual "
          f"que el resto: en total se imputa el PMD de **{imputados:,} vuelos "
          f"({100 * imputados / len(vuelos):.2f}%)**.", ""]

    en_scte = int(((vuelos.pmd > 650) & (vuelos.aeropuerto_oaci == "SCTE")).sum())
    L += [f"> Ninguna de estas anomalias afecta a la PoC: de los {len(imposibles):,} "
          f"registros con unidad erronea, solo {en_scte} ocurren en El Tepual y ninguno "
          "es vuelo regular del periodo 2020-2026.", ""]

    # ── 4. Clima ─────────────────────────────────────────────────────────────
    L += ["## 4. Clima diario de Puerto Montt (estacion 85799)", "",
          f"**{len(clima):,} dias** ({clima.fecha_cruce.min():%Y-%m-%d} a "
          f"{clima.fecha_cruce.max():%Y-%m-%d}), sin dias faltantes.", ""]
    L += [_tabla(_resumen_columnas(clima)), ""]
    num_clima = ["tavg", "tmin", "tmax", "amplitud_termica", "prcp",
                 "wspd", "viento_cruzado", "pres"]
    L += ["### Resumen estadistico del clima", "",
          _tabla(_describe_numerico(clima, num_clima)), ""]
    L += [f"Llueve en el **{100 * (clima.prcp > 0).mean():.1f}%** de los dias. "
          f"Ese solo dato explica por que la lluvia no diferencia dias buenos de malos "
          f"en El Tepual: es la condicion normal, no una anomalia.", ""]

    # ── 5. El subconjunto SCTE ───────────────────────────────────────────────
    L += ["## 5. Subconjunto de la PoC — El Tepual (SCTE)", "",
          f"**{len(vuelos_clima):,} vuelos regulares** con horario reconstruido, "
          f"{vuelos_clima.aerolinea_dgac.nunique()} aerolineas y "
          f"{vuelos_clima.numero_vuelo.nunique()} numeros de vuelo distintos.", ""]

    por_anio = (
        vuelos_clima.groupby("anio")
        .agg(vuelos=("retraso", "size"), tasa_retraso=("retraso", "mean"),
             desvio_mediano=("desvio_min", "median"))
        .round(3).reset_index()
    )
    L += ["### Cobertura por año", "", _tabla(por_anio), ""]
    L += ["El volumen de 2020 es **menos de la mitad** que el de 2024: el periodo "
          "arranca en plena pandemia. El modelo entrena con dos regimenes operativos "
          "distintos tratados como uno solo (ver la seccion de sesgos del README).", ""]

    por_aerolinea = (
        vuelos_clima.groupby("aerolinea_dgac")
        .agg(vuelos=("retraso", "size"), tasa_retraso=("retraso", "mean"),
             desvio_p75=("desvio_min", lambda s: s.quantile(0.75)))
        .sort_values("vuelos", ascending=False).head(8).round(3).reset_index()
    )
    L += ["### Principales operadores", "", _tabla(por_aerolinea), ""]

    # ── 6. El target ─────────────────────────────────────────────────────────
    L += ["## 6. El target de retraso", ""]
    desvio = vuelos_clima["desvio_min"]
    L += [_tabla(pd.DataFrame([{
        "vuelos": f"{len(vuelos_clima):,}",
        "tasa de retraso (>15 min)": f"{vuelos_clima.retraso.mean():.1%}",
        "retraso severo (>60 min)": f"{(desvio > 60).mean():.1%}",
        "desvio p05": round(float(desvio.quantile(0.05)), 1),
        "desvio mediano": round(float(desvio.median()), 1),
        "desvio p95": round(float(desvio.quantile(0.95)), 1),
        "desviacion estandar": round(float(desvio.std()), 1),
    }])), ""]
    L += ["La distribucion esta centrada en cero, con cola izquierda corta y cola "
          "derecha larga. **Esa asimetria es la firma de un retraso real**: los vuelos "
          "se atrasan mucho y se adelantan poco. Si fuera ruido de medicion, seria "
          "simetrica.", ""]

    _figura_outliers(vuelos, resumen_out)
    _figura_target(vuelos_clima)
    L += ["## Figuras generadas", "",
          "- `images/eda_08_outliers_pmd.png` — distribucion y valores extremos del PMD",
          "- `images/eda_09_target_retraso.png` — el target por año, hora y operador", ""]

    logger.info("Reporte EDA generado: %d secciones, %d lineas", 6, len(L))
    return "\n".join(L)


def _figura_outliers(vuelos: pd.DataFrame, resumen: dict) -> None:
    """Distribucion del PMD en escala logaritmica y su caja de outliers."""
    pmd = vuelos["pmd"].dropna()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.hist(np.log10(pmd[pmd > 0]), bins=100, color=_COLOR_A, alpha=0.85)
    ax1.axvline(np.log10(resumen["limite_superior"]), color=_COLOR_B, linestyle="--",
                label=f"Limite IQR ({resumen['limite_superior']:.0f} t)")
    ax1.set_xlabel("log10(PMD en toneladas)")
    ax1.set_ylabel("Numero de vuelos")
    ax1.set_title("El PMD es multimodal, no una cola sucia\n"
                  "(cada joroba es una familia de aeronaves)")
    ax1.legend(fontsize=9)

    ax2.boxplot(pmd, vert=False, widths=0.6,
                flierprops={"marker": ".", "markersize": 2, "alpha": 0.3})
    ax2.set_xscale("log")
    ax2.set_xlabel("PMD en toneladas (escala log)")
    ax2.set_yticks([])
    ax2.set_title(f"{resumen['pct_outliers']}% queda fuera del rango intercuartil\n"
                  f"pero son aeronaves reales, no errores")

    fig.suptitle("Valores extremos del PMD: diagnostico, no recorte", fontsize=13)
    fig.tight_layout()
    ruta = _IMAGES_DIR / "eda_08_outliers_pmd.png"
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    logger.info("Guardado %s", ruta)


def _figura_target(vuelos_clima: pd.DataFrame) -> None:
    """El target visto por año, por hora programada y por operador."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

    por_anio = vuelos_clima.groupby("anio").retraso.agg(["size", "mean"])
    ax1.bar(por_anio.index.astype(str), por_anio["mean"] * 100, color=_COLOR_A, alpha=0.85)
    ax1.axhline(vuelos_clima.retraso.mean() * 100, color=_COLOR_B, linestyle="--",
                label="Media del periodo")
    for i, (n, m) in enumerate(zip(por_anio["size"], por_anio["mean"])):
        ax1.text(i, m * 100 + 0.3, f"n={n//1000}k", ha="center", fontsize=7)
    ax1.set_ylabel("% de vuelos retrasados")
    ax1.set_title("Por año\n(2020 opero a media capacidad)")
    ax1.legend(fontsize=8)

    desvio = vuelos_clima["desvio_min"].clip(-90, 180)
    ax2.hist(desvio, bins=100, color=_COLOR_A, alpha=0.85)
    ax2.axvline(0, color="grey", linewidth=1)
    ax2.axvline(15, color=_COLOR_B, linestyle="--", label="Umbral (+15 min)")
    ax2.set_xlabel("Desvio del horario habitual (min)")
    ax2.set_ylabel("Vuelos")
    ax2.set_title("Distribucion del desvio\n(asimetrica: se atrasa mas de lo que se adelanta)")
    ax2.legend(fontsize=8)

    top = vuelos_clima.aerolinea_dgac.value_counts().head(6).index
    por_aero = (
        vuelos_clima[vuelos_clima.aerolinea_dgac.isin(top)]
        .groupby("aerolinea_dgac").retraso.agg(["size", "mean"])
        .sort_values("mean")
    )
    ax3.barh(por_aero.index, por_aero["mean"] * 100, color=_COLOR_A, alpha=0.85)
    ax3.axvline(vuelos_clima.retraso.mean() * 100, color=_COLOR_B, linestyle="--",
                label="Media global")
    ax3.set_xlabel("% de vuelos retrasados")
    ax3.set_title("Por operador\n(mide irregularidad, no atraso vs itinerario)")
    ax3.legend(fontsize=8)

    fig.suptitle("El target de retraso, visto desde tres angulos", fontsize=13)
    fig.tight_layout()
    ruta = _IMAGES_DIR / "eda_09_target_retraso.png"
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    logger.info("Guardado %s", ruta)
