"""
Nodos de preprocesamiento, integración y EDA de datos aeronáuticos (JAC).

Fuentes:
  - bitacora-vuelos.parquet : bitácora de vuelos (11M+ filas)
  - operaciones-aeropuertos.csv : operaciones agregadas por mes/aeropuerto/tipo
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

_IMAGES_DIR = Path("images")


def preprocess_vuelos(vuelos: pd.DataFrame) -> pd.DataFrame:
    """Limpia y enriquece la bitácora de vuelos para el JOIN posterior.

    Transformaciones aplicadas:
    - numero_vuelo : nulos → "DESCONOCIDO"
    - pmd          : flag de imputación + relleno por mediana grupal (modelo_avion)
    - mes_id       : extrae YYYYMM desde dt_operacion (zona America/Santiago)
    - internacional_domestico : mapea es_internacional (bool) → 'I' / 'D'
    """
    df = vuelos.copy()

    # numero_vuelo
    df["numero_vuelo"] = df["numero_vuelo"].fillna("DESCONOCIDO")

    # pmd: coerce a float, crear flag, imputar con mediana grupal
    df["pmd"] = pd.to_numeric(df["pmd"], errors="coerce")

    # Un avion no puede pesar cero: el 0 es un nulo disfrazado. Son ~26 mil
    # registros que llegan con el valor literal 0 en vez de vacio, asi que
    # imputar solo los NaN los dejaba pasar intactos al modelo.
    ceros = int((df["pmd"] == 0).sum())
    if ceros:
        logger.info("pmd = 0 en %d registros: se tratan como faltantes", ceros)
        df.loc[df["pmd"] == 0, "pmd"] = pd.NA
        df["pmd"] = pd.to_numeric(df["pmd"], errors="coerce")

    df["pmd_fue_imputado"] = df["pmd"].isna().astype("int8")

    global_median = df["pmd"].median()
    group_medians = df.groupby("modelo_avion")["pmd"].transform("median")
    df["pmd"] = df["pmd"].fillna(group_medians.fillna(global_median))

    # mes_id (YYYYMM)
    dt = df["dt_operacion"]
    if dt.dt.tz is not None:
        dt = dt.dt.tz_convert("America/Santiago")
    df["mes_id"] = (dt.dt.year * 100 + dt.dt.month).astype("int64")

    # internacional_domestico
    df["internacional_domestico"] = df["es_internacional"].map({True: "I", False: "D"})

    return df


def join_con_aeropuertos(
    vuelos_preprocessed: pd.DataFrame,
    aeropuertos: pd.DataFrame,
) -> pd.DataFrame:
    """LEFT JOIN entre vuelos preprocesados y operaciones por aeropuerto.

    Llaves: aeropuerto_oaci, mes_id, internacional_domestico.
    Incorpora la columna cnt_operaciones del CSV de aeropuertos.
    """
    cols_aeropuertos = ["aeropuerto_oaci", "mes_id", "internacional_domestico", "cnt_operaciones"]
    return pd.merge(
        vuelos_preprocessed,
        aeropuertos[cols_aeropuertos],
        on=["aeropuerto_oaci", "mes_id", "internacional_domestico"],
        how="left",
    )


def generate_eda_plots(vuelos: pd.DataFrame) -> pd.DataFrame:
    """Genera cuatro gráficos EDA y los guarda en images/.

    Gráficos producidos:
    1. eda_01_top_aeropuertos.png  — Top 15 aeropuertos por volumen de vuelos
    2. eda_02_intl_vs_dom.png      — Evolución anual: internacional vs doméstico
    3. eda_03_pmd_imputado.png     — Distribución de pmd: imputado vs original
    4. eda_04_operaciones_anio.png — Total de operaciones por año

    Devuelve un DataFrame con metadatos de los archivos generados.
    """
    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    PALETTE = {"D": "#4C72B0", "I": "#DD8452"}
    saved: list[dict] = []

    # ── 1. Top 15 aeropuertos por volumen ────────────────────────────────────
    top = (
        vuelos["aeropuerto_oaci"]
        .value_counts()
        .head(15)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.index, top.values, color="#4C72B0")
    ax.set_xlabel("Número de vuelos")
    ax.set_title("Top 15 aeropuertos por volumen de vuelos (1999-2026)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    fig.tight_layout()
    path = _IMAGES_DIR / "eda_01_top_aeropuertos.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append({"archivo": str(path), "descripcion": "Top 15 aeropuertos por volumen"})
    logger.info("Guardado %s", path)

    # ── 2. Internacional vs Doméstico por año ────────────────────────────────
    anio = vuelos["dt_operacion"].dt.year
    tabla = (
        vuelos.assign(anio=anio)
        .groupby(["anio", "internacional_domestico"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={"D": "Doméstico", "I": "Internacional"})
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    tabla.plot(kind="bar", stacked=True, ax=ax,
               color=[PALETTE["D"], PALETTE["I"]], width=0.8)
    ax.set_xlabel("Año")
    ax.set_ylabel("Número de vuelos")
    ax.set_title("Vuelos internacionales vs domésticos por año")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.legend(title="Tipo")
    fig.tight_layout()
    path = _IMAGES_DIR / "eda_02_intl_vs_dom.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append({"archivo": str(path), "descripcion": "Internacional vs Doméstico por año"})
    logger.info("Guardado %s", path)

    # ── 3. PMD: imputado vs original ─────────────────────────────────────────
    pmd_orig = vuelos.loc[vuelos["pmd_fue_imputado"] == 0, "pmd"].clip(upper=500)
    pmd_imp  = vuelos.loc[vuelos["pmd_fue_imputado"] == 1, "pmd"].clip(upper=500)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pmd_orig, bins=80, alpha=0.6, label="Original", color="#4C72B0", density=True)
    ax.hist(pmd_imp,  bins=80, alpha=0.6, label="Imputado", color="#DD8452", density=True)
    ax.set_xlabel("pmd (truncado en 500)")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución de pmd: valores originales vs imputados")
    ax.legend()
    fig.tight_layout()
    path = _IMAGES_DIR / "eda_03_pmd_imputado.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append({"archivo": str(path), "descripcion": "Distribución pmd imputado vs original"})
    logger.info("Guardado %s", path)

    # ── 4. Total operaciones por año ─────────────────────────────────────────
    ops_anio = (
        vuelos.assign(anio=vuelos["dt_operacion"].dt.year)
        .groupby("anio")
        .size()
        .reset_index(name="vuelos")
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ops_anio["anio"], ops_anio["vuelos"], marker="o", linewidth=2, color="#4C72B0")
    ax.fill_between(ops_anio["anio"], ops_anio["vuelos"], alpha=0.15, color="#4C72B0")
    ax.set_xlabel("Año")
    ax.set_ylabel("Número de vuelos")
    ax.set_title("Evolución anual de operaciones aéreas en Chile")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    fig.tight_layout()
    path = _IMAGES_DIR / "eda_04_operaciones_anio.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append({"archivo": str(path), "descripcion": "Evolución anual de operaciones"})
    logger.info("Guardado %s", path)

    return pd.DataFrame(saved)
