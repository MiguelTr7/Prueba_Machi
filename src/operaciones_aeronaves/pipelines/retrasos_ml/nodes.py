"""
Nodos del PoC de riesgo de retraso en SCTE (El Tepual, Puerto Montt).

Pipeline paralelo al baseline: no toca `data_processing` ni `ml`. Responde una
pregunta de negocio distinta — *cuanto puede el clima retrasar un vuelo* — y
entrega un Score de Riesgo Operativo (probabilidad 0-100%) usable como proxy de
costo operacional.

ADVERTENCIA SOBRE EL TARGET
---------------------------
La bitacora JAC tiene una sola marca de tiempo (`dt_operacion`, la hora **real**
de la operacion). No publica hora programada, de modo que el retraso no se puede
calcular restando "programada - real" como se planteo originalmente. Aqui el
horario de referencia se **reconstruye** desde la propia bitacora: la mediana
circular del slot habitual de cada vuelo. El detalle y sus limites estan en el
README, seccion "Como se construye el retraso si el dato no lo trae".

Entradas esperadas:
  - vuelos_raw          : bitacora JAC completa (11M filas, data/01_raw)
  - clima_diario_raw    : clima diario Meteostat estacion 85799 (data/01_raw)
Salidas en catalogo:
  - vuelos_scte_target, clima_scte_features, vuelos_clima
  - metricas_retrasos, modelo_retrasos, retrasos_figura_manifest
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

_IMAGES_DIR = Path("images")

_COLOR_A = "#4C72B0"
_COLOR_B = "#DD8452"


def _envolver(minutos):
    """Lleva una diferencia de minutos al rango [-720, 720).

    Un vuelo de las 23:50 y otro de las 00:10 distan 20 minutos, no 1420. Sin
    esta correccion los vuelos que cruzan medianoche destruyen cualquier medida
    de dispersion horaria.
    """
    return ((minutos + 720) % 1440) - 720


# ─────────────────────────────────────────────────────────────────────────────
# 1. RECORTE DE LA PoC
# ─────────────────────────────────────────────────────────────────────────────

def filtrar_vuelos_scte(vuelos: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Recorta la bitacora al aeropuerto, periodo y actividad de la PoC.

    De 11 millones de filas quedan ~76 mil. Es lo que permite correr el pipeline
    completo en un notebook sin agotar la RAM; escalar a todo Chile queda como
    trabajo futuro (ver README, "Trabajo futuro").
    """
    df = vuelos[vuelos["aeropuerto_oaci"] == params["aeropuerto_oaci"]].copy()

    desde = pd.Timestamp(params["fecha_desde"], tz="America/Santiago")
    hasta = pd.Timestamp(params["fecha_hasta"], tz="America/Santiago")
    df = df[(df["dt_operacion"] >= desde) & (df["dt_operacion"] < hasta)]

    # Solo vuelos con itinerario: sin itinerario no hay "retraso" que medir.
    df = df[df["actividad_cod"].isin(params["actividades"])]

    logger.info(
        "Filtro PoC %s %s..%s: %d vuelos regulares",
        params["aeropuerto_oaci"], params["fecha_desde"], params["fecha_hasta"], len(df),
    )
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. RECONSTRUCCION DEL HORARIO Y TARGET DE RETRASO
# ─────────────────────────────────────────────────────────────────────────────

def _referencia_por_slots(minutos: pd.Series, gap: int, min_obs: int) -> pd.Series:
    """Horario de referencia de un grupo de operaciones del mismo vuelo.

    Un mismo numero de vuelo puede tener mas de un slot habitual (LAN 61 llega
    ~07:16 los sabados y ~19:34 lunes y miercoles). Promediar ambos daria un
    horario que no existe, asi que primero se separan los slots por huecos
    mayores a `gap` minutos sobre el reloj circular, y despues cada operacion se
    compara contra el centro del slot mas cercano.
    """
    orden = np.sort(minutos.values)
    if len(orden) == 1:
        return pd.Series(orden[0], index=minutos.index)

    # Huecos sobre el circulo de 24 h (el ultimo cierra con el primero).
    huecos = np.diff(np.append(orden, orden[0] + 1440))
    cortes = np.where(huecos > gap)[0]

    if len(cortes) == 0:
        grupos = [orden]
    else:
        # Rotar para que el arreglo empiece justo despues del hueco mas tardio:
        # asi un slot que cruza medianoche queda contiguo.
        inicio = (cortes[-1] + 1) % len(orden)
        rotado = np.roll(orden, -inicio)
        limites = np.sort((cortes - inicio) % len(orden))
        grupos = np.split(rotado, limites[:-1] + 1) if len(limites) > 1 else [rotado]

    centros, pesos = [], []
    for grupo in grupos:
        base = grupo[0]
        centros.append((base + np.median(_envolver(grupo - base))) % 1440)
        pesos.append(len(grupo))

    centros, pesos = np.array(centros), np.array(pesos)

    # Un slot con muy pocas observaciones suele ser el propio retraso que
    # queremos medir; se descarta como "horario habitual" para que esas
    # operaciones se midan contra el slot grande y el retraso no se auto-anule.
    habituales = centros[pesos >= min_obs]
    if len(habituales) == 0:
        habituales = centros[[np.argmax(pesos)]]

    distancia = np.abs(_envolver(minutos.values[:, None] - habituales[None, :]))
    return pd.Series(habituales[np.argmin(distancia, axis=1)], index=minutos.index)


def construir_target_retraso(vuelos_scte: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Reconstruye el horario habitual de cada vuelo y deriva el target binario.

    - `hora_referencia_min` : horario habitual reconstruido (minutos del dia)
    - `desvio_min`          : minutos por sobre (+) o bajo (-) ese horario
    - `retraso`             : 1 si `desvio_min` > umbral (15 min), si no 0
    - `fecha_cruce`         : YYYY-MM-DD, la llave contra el clima diario

    La agrupacion incluye el dia de la semana porque las aerolineas programan
    por dia: sin el, el slot de sabado y el de lunes se mezclan y la dispersion
    se dispara (std 168 min vs 72 min).
    """
    df = vuelos_scte.copy()

    df["min_dia"] = df["dt_operacion"].dt.hour * 60 + df["dt_operacion"].dt.minute
    df["anio"] = df["dt_operacion"].dt.year
    df["mes"] = df["dt_operacion"].dt.month
    df["dia_semana"] = df["dt_operacion"].dt.dayofweek
    df["mes_ym"] = df["dt_operacion"].dt.strftime("%Y-%m")

    llave = ["aerolinea_dgac", "numero_vuelo", "tipo_operacion", "mes_ym", "dia_semana"]
    agrupado = df.groupby(llave)["min_dia"]

    df["n_obs_grupo"] = agrupado.transform("size")
    df["hora_referencia_min"] = agrupado.transform(
        _referencia_por_slots,
        gap=params["slot_gap_min"],
        min_obs=params["slot_min_obs"],
    )
    df["desvio_min"] = _envolver(df["min_dia"] - df["hora_referencia_min"])

    # Grupos con muy pocas observaciones no permiten estimar un horario habitual
    # confiable: su "retraso" seria ruido, asi que se excluyen del modelo.
    antes = len(df)
    df = df[df["n_obs_grupo"] >= params["grupo_min_obs"]].copy()

    df["retraso"] = (df["desvio_min"] > params["umbral_retraso_min"]).astype("int8")
    df["hora_programada"] = df["hora_referencia_min"] / 60.0

    # Llave de cruce con el clima: se elimina la hora y la zona horaria para que
    # el merge sea un left join limpio de un solo campo.
    df["fecha_cruce"] = df["dt_operacion"].dt.tz_localize(None).dt.normalize()

    logger.info(
        "Target construido: %d vuelos (%.1f%% del recorte) | tasa de retraso %.3f",
        len(df), 100 * len(df) / antes, df["retraso"].mean(),
    )
    logger.info(
        "Desvio (min) p05=%.0f p50=%.0f p95=%.0f — la cola izquierda debe ser "
        "corta: un vuelo no sale horas antes de lo habitual",
        df["desvio_min"].quantile(0.05), df["desvio_min"].median(),
        df["desvio_min"].quantile(0.95),
    )
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLIMA
# ─────────────────────────────────────────────────────────────────────────────

def preparar_clima_scte(clima: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Limpia el clima diario de Meteostat y deriva las variables operativas.

    `snow`, `wpgt` y `tsun` vienen 100% vacias para esta estacion y se descartan.
    Se agrega el **viento cruzado**, que es la variable que de verdad limita la
    operacion: un viento de 40 km/h alineado con la pista casi no molesta, el
    mismo viento de costado si.
    """
    df = clima.copy()
    df["fecha_cruce"] = pd.to_datetime(df["fecha"])

    vacias = [c for c in ("snow", "wpgt", "tsun") if c in df.columns and df[c].isna().all()]
    if vacias:
        logger.info("Columnas de clima 100%% vacias, descartadas: %s", ", ".join(vacias))
        df = df.drop(columns=vacias)

    # Sin registro de lluvia = no llovio. Es la convencion de Meteostat y evita
    # perder el 18% de los dias en el modelo.
    df["prcp"] = df["prcp"].fillna(0.0)
    df["amplitud_termica"] = df["tmax"] - df["tmin"]

    # Descomposicion del viento respecto al rumbo de la pista 17/35.
    rumbo = np.radians(df["wdir"] - params["rumbo_pista"])
    df["viento_cruzado"] = (df["wspd"] * np.sin(rumbo).abs()).fillna(df["wspd"] * 0.5)
    df["viento_frontal"] = (df["wspd"] * np.cos(rumbo)).fillna(0.0)

    df["dia_lluvioso"] = (df["prcp"] > 1.0).astype("int8")

    columnas = ["fecha_cruce", "tavg", "tmin", "tmax", "amplitud_termica", "prcp",
                "dia_lluvioso", "wspd", "wdir", "viento_cruzado", "viento_frontal", "pres"]
    df = df[columnas]

    logger.info("Clima preparado: %d dias (%s a %s)",
                len(df), df.fecha_cruce.min().date(), df.fecha_cruce.max().date())
    return df


def cruzar_vuelos_clima(vuelos: pd.DataFrame, clima: pd.DataFrame) -> pd.DataFrame:
    """LEFT JOIN vuelos x clima por `fecha_cruce` (un solo campo YYYY-MM-DD)."""
    salida = pd.merge(vuelos, clima, on="fecha_cruce", how="left")

    sin_clima = salida["tavg"].isna().mean()
    logger.info("Cruce: %d vuelos | %.2f%% sin clima", len(salida), 100 * sin_clima)
    if sin_clima > 0.05:
        logger.warning(
            "Mas del 5%% de los vuelos quedaron sin clima. Revisa que el CSV "
            "cubra todo el periodo (el bulk de Meteostat corta ~1 mes atras)."
        )
    return salida


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODELO — SCORE DE RIESGO OPERATIVO
# ─────────────────────────────────────────────────────────────────────────────

_FEATURES_CLIMA = ["tavg", "tmin", "tmax", "amplitud_termica", "prcp", "dia_lluvioso",
                   "wspd", "viento_cruzado", "viento_frontal", "pres"]
_FEATURES_OPERA = ["hora_programada", "dia_semana", "mes", "anio",
                   "aerolinea_cod", "tipo_operacion_cod", "modelo_cod", "es_internacional"]


def _matriz_features(df: pd.DataFrame) -> pd.DataFrame:
    """Arma la matriz de features. NO usa la hora real: eso seria fuga del target.

    `hora_programada` es el horario habitual reconstruido — el dato que un
    planificador si conoce con dias de anticipacion. La hora real (`min_dia`) es
    justamente lo que el modelo debe predecir y queda fuera.
    """
    X = pd.DataFrame(index=df.index)
    for col in _FEATURES_CLIMA:
        X[col] = pd.to_numeric(df[col], errors="coerce")
    X["hora_programada"] = df["hora_programada"]
    X["dia_semana"] = df["dia_semana"]
    X["mes"] = df["mes"]
    X["anio"] = df["anio"]
    X["aerolinea_cod"] = df["aerolinea_dgac"].astype("category").cat.codes
    X["tipo_operacion_cod"] = (df["tipo_operacion"] == "D").astype("int8")
    X["modelo_cod"] = df["modelo_avion"].astype("category").cat.codes
    X["es_internacional"] = df["es_internacional"].astype("int8")
    return X[_FEATURES_CLIMA + _FEATURES_OPERA]


def _grilla(params: dict) -> dict[str, list[dict]]:
    """Configuraciones candidatas de cada algoritmo.

    **Por que estos dos algoritmos.** El problema es una clasificacion binaria
    desbalanceada (15% de positivos) con variables mixtas: gradient boosting
    captura interacciones no lineales entre hora, aerolinea y clima sin pedir
    escalado ni codificacion previa, y es el estandar para datos tabulares.
    La regresion logistica entra como contraparte deliberadamente simple: si un
    modelo lineal iguala al boosting, es señal de que no hay estructura no
    lineal que aprender — y eso es exactamente lo que termino ocurriendo.
    Ademas entrega probabilidades calibradas de fabrica, que es lo que necesita
    un score leido como porcentaje.

    **Por que la grilla es chica y sesgada a la regularizacion.** Con una señal
    debil, el boosting sin podar memoriza: llega a AUC 0.84 en entrenamiento y
    0.52 en validacion. La grilla recorre desde configuraciones flexibles hasta
    muy restringidas para que la eleccion del punto sea del dato, no nuestra.

    Las configuraciones viven en `conf/base/parameters_retrasos_ml.yml`: probar
    otra grilla no deberia exigir tocar el codigo.
    """
    semilla = params["random_state"]
    return {
        "GradientBoosting": [
            {**config, "random_state": semilla, "early_stopping": False}
            for config in params["grilla_gradient_boosting"]
        ],
        "RegresionLogistica": [
            {**config, "max_iter": 1000, "random_state": semilla}
            for config in params["grilla_logistica"]
        ],
    }


def _construir(algoritmo: str, config: dict):
    """Instancia un estimador a partir de su configuracion."""
    if algoritmo == "GradientBoosting":
        return HistGradientBoostingClassifier(**config)
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(**config),
    )


def _metricas_test(nombre: str, conjunto: str, y_test, proba) -> dict:
    tasa_base = float(np.mean(y_test))
    ap = average_precision_score(y_test, proba)
    pred = (proba >= 0.5).astype(int)
    return {
        "modelo": nombre,
        "features": conjunto,
        "tasa_base_test": round(tasa_base, 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "average_precision": round(ap, 4),
        "lift_vs_azar": round(ap / tasa_base, 3),
        "brier": round(brier_score_loss(y_test, proba), 4),
        "precision_u050": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall_u050": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1_u050": round(f1_score(y_test, pred, zero_division=0), 4),
    }


def train_modelo_retrasos(vuelos_clima: pd.DataFrame, params: dict) -> tuple:
    """Entrena el Score de Riesgo Operativo con split TEMPORAL en tres tramos.

    El modelo se vende como pronostico a 7-14 dias, asi que jamas se evalua con
    un split aleatorio: eso pondria vuelos del mismo dia en train y en test, y
    las metricas saldrian infladas. El corte es:

        train  : anios anteriores a `anio_validacion`
        val    : `anio_validacion`  — elige modelo e hiperparametros
        test   : `anio_corte_test` en adelante — se toca una sola vez

    Ademas entrena tres veces cada candidato — solo operacion, solo clima, y
    ambos — porque esa comparacion *es* la respuesta a la pregunta del proyecto:
    si sumar el clima no mueve el AUC, el clima no explica el retraso.

    Devuelve (metricas por modelo, tabla de la busqueda, artefacto del ganador).
    """
    df = vuelos_clima.dropna(subset=["tavg"]).copy()

    X = _matriz_features(df)
    y = df["retraso"].astype(int).values

    anio_val = params["anio_validacion"]
    corte = params["anio_corte_test"]
    es_train = (df["anio"] < anio_val).values
    es_val = (df["anio"] == anio_val).values
    es_test = (df["anio"] >= corte).values

    logger.info("Split temporal: train %d (<%d) | val %d (%d) | test %d (>=%d)",
                es_train.sum(), anio_val, es_val.sum(), anio_val, es_test.sum(), corte)

    conjuntos = {
        "solo operacion": _FEATURES_OPERA,
        "solo clima": _FEATURES_CLIMA,
        "clima + operacion": _FEATURES_CLIMA + _FEATURES_OPERA,
    }

    # Busqueda de hiperparametros: cada configuracion se ajusta con el tramo de
    # entrenamiento y se puntua contra el año de validacion. El tramo de prueba
    # no participa en ninguna decision.
    filas, ajustados, busqueda = [], {}, []
    for conjunto, columnas in conjuntos.items():
        Xc = X[columnas]
        for algoritmo, configuraciones in _grilla(params).items():
            mejor = None
            for config in configuraciones:
                modelo = _construir(algoritmo, config)
                modelo.fit(Xc[es_train], y[es_train])
                auc_val = roc_auc_score(y[es_val], modelo.predict_proba(Xc[es_val])[:, 1])
                auc_train = roc_auc_score(y[es_train], modelo.predict_proba(Xc[es_train])[:, 1])

                busqueda.append({
                    "features": conjunto,
                    "algoritmo": algoritmo,
                    "config": ", ".join(
                        f"{k}={v}" for k, v in config.items() if k != "random_state"
                    ),
                    "roc_auc_train": round(auc_train, 4),
                    "roc_auc_val": round(auc_val, 4),
                    # La brecha train-val delata la memorizacion.
                    "brecha_sobreajuste": round(auc_train - auc_val, 4),
                })
                if mejor is None or auc_val > mejor[1]:
                    mejor = (modelo, auc_val, auc_train, config)

            modelo, auc_val, auc_train, config = mejor
            proba_test = modelo.predict_proba(Xc[es_test])[:, 1]

            fila = _metricas_test(algoritmo, conjunto, y[es_test], proba_test)
            fila["roc_auc_val"] = round(auc_val, 4)
            fila["roc_auc_train"] = round(auc_train, 4)
            fila["config"] = ", ".join(f"{k}={v}" for k, v in config.items() if k != "random_state")
            filas.append(fila)
            ajustados[(conjunto, algoritmo)] = (modelo, columnas, proba_test)

    busqueda = pd.DataFrame(busqueda)
    logger.info("Busqueda de hiperparametros: %d configuraciones evaluadas", len(busqueda))
    peor = busqueda.loc[busqueda.brecha_sobreajuste.idxmax()]
    logger.info(
        "Mayor sobreajuste observado: %s (%s) — AUC train %.4f vs val %.4f",
        peor.algoritmo, peor.config, peor.roc_auc_train, peor.roc_auc_val,
    )

    metricas = pd.DataFrame(filas)

    # El ganador se elige por el anio de validacion sobre TODAS las
    # combinaciones — incluido el conjunto sin clima. Si el clima no aporta, el
    # propio modelo entregado lo deja en evidencia al no incluirlo.
    ganador = metricas.loc[metricas.roc_auc_val.idxmax()]
    modelo, columnas, proba = ajustados[(ganador.features, ganador.modelo)]
    metricas["seleccionado"] = (
        (metricas.modelo == ganador.modelo) & (metricas.features == ganador.features)
    )

    logger.info("Modelo elegido por validacion %d: %s sobre '%s' | AUC val %.4f -> test %.4f",
                anio_val, ganador.modelo, ganador.features, ganador.roc_auc_val, ganador.roc_auc)
    logger.info("Comparacion de conjuntos de features (ROC-AUC en test):\n%s",
                metricas.pivot_table(index="features", columns="modelo",
                                     values="roc_auc").to_string())

    # Importancia por permutacion: a diferencia del Gini, no premia a las
    # variables con muchos valores distintos, asi que la comparacion
    # clima vs operacion es justa. Se mide sobre el modelo COMPLETO aunque el
    # ganador no lo sea: es el unico que tiene ambos bloques que comparar.
    modelo_full, columnas_full, _ = ajustados[("clima + operacion", ganador.modelo)]
    perm = permutation_importance(
        modelo_full, X[columnas_full][es_test], y[es_test], n_repeats=5,
        random_state=params["random_state"], scoring="roc_auc", n_jobs=-1,
    )
    importancia = (
        pd.DataFrame({"feature": columnas_full,
                      "importancia": perm.importances_mean,
                      "std": perm.importances_std})
        .assign(bloque=lambda d: np.where(d.feature.isin(_FEATURES_CLIMA), "Clima", "Operacion"))
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )

    artefacto = {
        "modelo": modelo,
        "nombre_modelo": ganador.modelo,
        "busqueda_hiperparametros": busqueda,
        "conjunto_features": ganador.features,
        "features": columnas,
        "features_clima": _FEATURES_CLIMA,
        "importancia": importancia,
        "y_test": y[es_test],
        "proba_test": proba,
        "anio_validacion": anio_val,
        "anio_corte_test": corte,
        "umbral_retraso_min": params["umbral_retraso_min"],
    }
    return metricas, busqueda, artefacto


# ─────────────────────────────────────────────────────────────────────────────
# 5. ¿CUANTO RETRASA REALMENTE EL CLIMA?
# ─────────────────────────────────────────────────────────────────────────────

_CORTES_LLUVIA = [-0.1, 0, 1, 5, 15, 1000]
_ETIQ_LLUVIA = ["Sin lluvia", "0-1 mm", "1-5 mm", "5-15 mm", "> 15 mm"]
_CORTES_VIENTO = [-0.1, 5, 10, 20, 30, 100]
_ETIQ_VIENTO = ["< 5", "5-10", "10-20", "20-30", "> 30"]


def _grafico_busqueda(busqueda: pd.DataFrame) -> None:
    """Muestra como la regularizacion cierra la brecha entre train y validacion.

    Es la evidencia que justifica la configuracion elegida: la version flexible
    del boosting alcanza un AUC altisimo en entrenamiento y no lo sostiene en
    validacion. Esa brecha es memorizacion, no aprendizaje.
    """
    gb = busqueda[
        (busqueda.algoritmo == "GradientBoosting")
        & (busqueda.features == "clima + operacion")
    ].reset_index(drop=True)
    if gb.empty:
        return

    def _valor(config: str, clave: str) -> str:
        """Extrae 'clave=valor' de la cadena de configuracion."""
        return config.split(f"{clave}=")[1].split(",")[0]

    # Las configuraciones ya vienen de mas flexible a mas restringida.
    etiquetas = [
        "prof={}\nhojas={}".format(_valor(c, "max_depth"), _valor(c, "min_samples_leaf"))
        for c in gb.config
    ]
    x = np.arange(len(gb))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(x, gb.roc_auc_train, marker="o", linewidth=2, color=_COLOR_B, label="Entrenamiento")
    ax1.plot(x, gb.roc_auc_val, marker="o", linewidth=2, color=_COLOR_A, label="Validacion (2024)")
    ax1.fill_between(x, gb.roc_auc_val, gb.roc_auc_train, alpha=0.12, color=_COLOR_B)
    ax1.axhline(0.5, color="grey", linestyle=":", linewidth=1, label="Azar")
    ax1.set_xticks(x)
    ax1.set_xticklabels(etiquetas, fontsize=8)
    ax1.set_xlabel("← mas flexible          Configuracion          mas regularizado →")
    ax1.set_ylabel("ROC-AUC")
    ax1.set_title("El area sombreada es memorizacion, no aprendizaje")
    ax1.legend(fontsize=9)

    ax2.bar(x, gb.brecha_sobreajuste, color=_COLOR_B, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(etiquetas, fontsize=8)
    ax2.set_xlabel("← mas flexible          Configuracion          mas regularizado →")
    ax2.set_ylabel("AUC entrenamiento − AUC validacion")
    ax2.set_title("Brecha de sobreajuste por configuracion")

    fig.suptitle("Por que el modelo final esta fuertemente regularizado", fontsize=13)
    fig.tight_layout()
    ruta = _IMAGES_DIR / "rt_08_busqueda_hiperparametros.png"
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    logger.info("Guardado %s", ruta)


def evaluar_impacto_clima(vuelos_clima: pd.DataFrame, artefacto: dict) -> pd.DataFrame:
    """Cuantifica el efecto del clima y genera los cuatro graficos del PoC.

    rt_01_clima_vs_retraso.png  — tasa de retraso por lluvia y viento cruzado
    rt_02_importancia.png       — importancia por permutacion, clima vs operacion
    rt_03_hora_del_dia.png      — propagacion del retraso a lo largo del dia
    rt_04_roc_calibracion.png   — ROC y calibracion del Score de Riesgo

    Devuelve la tabla de efecto del clima (tasa de retraso por rango), que es la
    respuesta numerica a "hasta que punto el clima puede retrasar un vuelo".
    """
    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    df = vuelos_clima.dropna(subset=["tavg"]).copy()
    tasa_base = df["retraso"].mean()

    tabla_lluvia = df.groupby(
        pd.cut(df.prcp, _CORTES_LLUVIA, labels=_ETIQ_LLUVIA), observed=True
    )["retraso"].agg(["size", "mean"])
    tabla_viento = df.groupby(
        pd.cut(df.viento_cruzado, _CORTES_VIENTO, labels=_ETIQ_VIENTO), observed=True
    )["retraso"].agg(["size", "mean"])

    # ── 1. Clima vs retraso ──────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, tabla, titulo, xlabel in [
        (ax1, tabla_lluvia, "Retraso segun lluvia del dia", "Lluvia acumulada del dia"),
        (ax2, tabla_viento, "Retraso segun viento cruzado",
         "Viento cruzado a la pista 17/35 (km/h)"),
    ]:
        ax.bar(tabla.index.astype(str), tabla["mean"] * 100, color=_COLOR_A, alpha=0.85)
        ax.axhline(tasa_base * 100, color=_COLOR_B, linestyle="--",
                   label=f"Tasa base {tasa_base*100:.1f}%")
        for i, (n, m) in enumerate(zip(tabla["size"], tabla["mean"])):
            ax.text(i, m * 100 + 0.4, f"n={n:,}", ha="center", fontsize=7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("% de vuelos retrasados")
        ax.set_title(titulo)
        # Holgura arriba para que las etiquetas n= no choquen con la leyenda.
        ax.set_ylim(0, tabla["mean"].max() * 100 * 1.25)
        ax.legend(fontsize=8, loc="lower right")
    ax1.tick_params(axis="x", labelrotation=20)

    fig.suptitle("¿Cuanto retrasa el clima un vuelo en El Tepual?", fontsize=13)
    fig.tight_layout()
    ruta = _IMAGES_DIR / "rt_01_clima_vs_retraso.png"
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    logger.info("Guardado %s", ruta)

    # ── 2. Importancia: clima vs operacion ───────────────────────────────────
    imp = artefacto["importancia"].sort_values("importancia")
    colores = [_COLOR_B if b == "Clima" else _COLOR_A for b in imp["bloque"]]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp["feature"], imp["importancia"], xerr=imp["std"], color=colores, alpha=0.9)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Caida de ROC-AUC al permutar la variable")
    ax.set_title("Que mueve realmente el riesgo de retraso\n"
                 "(naranjo = clima, azul = operacion)")
    fig.tight_layout()
    ruta = _IMAGES_DIR / "rt_02_importancia.png"
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    logger.info("Guardado %s", ruta)

    # ── 3. Propagacion del retraso durante el dia ────────────────────────────
    por_hora = df.groupby(df["hora_programada"].astype(int))["retraso"].agg(["size", "mean"])
    por_hora = por_hora[por_hora["size"] >= 100]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(por_hora.index, por_hora["mean"] * 100, marker="o", linewidth=2, color=_COLOR_A)
    ax.fill_between(por_hora.index, por_hora["mean"] * 100, alpha=0.15, color=_COLOR_A)
    ax.axhline(tasa_base * 100, color=_COLOR_B, linestyle="--",
               label=f"Tasa base {tasa_base*100:.1f}%")
    ax.set_xlabel("Hora programada de la operacion")
    ax.set_ylabel("% de vuelos retrasados")
    ax.set_title("El retraso se acumula durante el dia\n"
                 "(un vuelo de la tarde hereda los atrasos de la manana)")
    ax.legend()
    fig.tight_layout()
    ruta = _IMAGES_DIR / "rt_03_hora_del_dia.png"
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    logger.info("Guardado %s", ruta)

    # ── 4. ROC + Precision-Recall + calibracion ──────────────────────────────
    y_test, proba = artefacto["y_test"], artefacto["proba_test"]
    tasa_base_test = float(np.mean(y_test))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax1.plot(fpr, tpr, color=_COLOR_A, linewidth=2, label=f"Score de Riesgo (AUC = {auc:.3f})")
    ax1.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="Azar (AUC = 0.5)")
    ax1.fill_between(fpr, tpr, alpha=0.08, color=_COLOR_A)
    ax1.set_xlabel("Falsos positivos")
    ax1.set_ylabel("Verdaderos positivos")
    ax1.set_title(f"Curva ROC — {artefacto['nombre_modelo']}\ntest {artefacto['anio_corte_test']}+")
    ax1.legend(loc="lower right", fontsize=9)

    # Precision-Recall: con solo 14% de positivos, la ROC se ve mejor de lo que
    # el modelo es. La PR compara contra la tasa base, que es la linea honesta.
    precision, recall, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    ax2.plot(recall, precision, color=_COLOR_A, linewidth=2, label=f"Score de Riesgo (AP = {ap:.3f})")
    ax2.axhline(tasa_base_test, color=_COLOR_B, linestyle="--",
                label=f"Tasa base ({tasa_base_test:.3f})")
    ax2.set_xlabel("Recall — % de retrasos detectados")
    ax2.set_ylabel("Precision — % de aciertos entre los avisados")
    ax2.set_ylim(0, max(0.5, precision.max() * 1.1))
    ax2.set_title("Curva Precision-Recall\n(la metrica correcta con clases desbalanceadas)")
    ax2.legend(fontsize=9)

    # Calibracion: el score solo sirve como "probabilidad" si un score de 30%
    # corresponde de verdad a un 30% de vuelos retrasados.
    bins = pd.qcut(proba, 10, duplicates="drop")
    cal = pd.DataFrame({"p": proba, "y": y_test}).groupby(bins, observed=True).mean()
    ax3.plot(cal["p"], cal["y"], marker="o", linewidth=2, color=_COLOR_A, label="Score observado")
    limite = float(cal["p"].max())
    ax3.plot([0, limite], [0, limite], linestyle="--", color="grey",
             label="Calibracion perfecta")
    ax3.set_xlabel("Score de Riesgo predicho")
    ax3.set_ylabel("Retraso real observado")
    ax3.set_title("Calibracion del Score de Riesgo")
    ax3.legend(fontsize=9)

    fig.tight_layout()
    ruta = _IMAGES_DIR / "rt_04_roc_calibracion.png"
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    logger.info("Guardado %s", ruta)

    # ── 5. La busqueda de hiperparametros, como evidencia ────────────────────
    _grafico_busqueda(artefacto["busqueda_hiperparametros"])

    # ── Tabla de efecto del clima, para citar en el README ───────────────────
    filas = []
    for nombre, tabla in [("lluvia_mm", tabla_lluvia), ("viento_cruzado_kmh", tabla_viento)]:
        for etiqueta, fila in tabla.iterrows():
            filas.append({
                "variable": nombre,
                "rango": str(etiqueta),
                "n_vuelos": int(fila["size"]),
                "tasa_retraso": round(float(fila["mean"]), 4),
                "puntos_vs_base": round(float(fila["mean"] - tasa_base) * 100, 2),
            })
    efecto = pd.DataFrame(filas)
    logger.info("Efecto del clima sobre el retraso:\n%s", efecto.to_string(index=False))
    return efecto
