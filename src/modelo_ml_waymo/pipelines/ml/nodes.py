"""
Nodos de EDA avanzado, clustering no supervisado y clasificación supervisada.

Entradas esperadas:
  - vuelos_con_operaciones : parquet de salida del pipeline data_processing
Salidas en catálogo:
  - aeropuertos_perfil, cluster_labels, metricas_supervisado
  - imágenes en images/ (eda_05..07, ml_01..04)
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_curve, auc as sklearn_auc
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

_IMAGES_DIR = Path("images")
_SAMPLE_SIZE = 500_000
_RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# EDA AVANZADO
# ─────────────────────────────────────────────────────────────────────────────

def perfil_aeropuertos(vuelos: pd.DataFrame) -> pd.DataFrame:
    """Agrega métricas operativas por aeropuerto para clustering y EDA."""
    return (
        vuelos.groupby("aeropuerto_oaci")
        .agg(
            total_vuelos=("aeropuerto_oaci", "count"),
            pct_intl=("es_internacional", "mean"),
            pmd_mediana=("pmd", "median"),
            n_aerolineas=("aerolinea_dgac", "nunique"),
            n_modelos=("modelo_avion", "nunique"),
            cnt_ops_media=("cnt_operaciones", "mean"),
        )
        .reset_index()
    )


def eda_avanzado(vuelos: pd.DataFrame, perfil: pd.DataFrame) -> pd.DataFrame:
    """Genera tres gráficos EDA adicionales y los guarda en images/.

    Gráficos:
    eda_05_pmd_vs_intl_aeropuerto.png — burbuja: PMD mediana vs % internacional por aeropuerto
    eda_06_heatmap_top12.png          — heatmap vuelos Top-12 aeropuertos × año
    eda_07_correlacion_numericas.png  — correlación entre variables numéricas de vuelo
    """
    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[dict] = []

    # ── 5. Burbuja PMD vs % internacional por aeropuerto ─────────────────────
    fig, ax = plt.subplots(figsize=(11, 7))
    sizes = np.sqrt(perfil["total_vuelos"]) / 8
    scatter = ax.scatter(
        np.log1p(perfil["pmd_mediana"]),
        perfil["pct_intl"] * 100,
        s=sizes,
        alpha=0.7,
        c=perfil["n_aerolineas"],
        cmap="viridis",
        edgecolors="grey",
        linewidth=0.4,
    )
    for _, row in perfil[perfil["total_vuelos"] > 50_000].iterrows():
        ax.annotate(
            row["aeropuerto_oaci"],
            (np.log1p(row["pmd_mediana"]), row["pct_intl"] * 100),
            fontsize=7,
            alpha=0.85,
        )
    plt.colorbar(scatter, ax=ax, label="Número de aerolíneas distintas")
    ax.set_xlabel("log(1 + PMD mediana) [toneladas]")
    ax.set_ylabel("% vuelos internacionales")
    ax.set_title("PMD mediana vs % internacional por aeropuerto\n(tamano ~ sqrt(volumen))")
    fig.tight_layout()
    path = _IMAGES_DIR / "eda_05_pmd_vs_intl_aeropuerto.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append({"archivo": str(path), "descripcion": "PMD vs % internacional por aeropuerto"})
    logger.info("Guardado %s", path)

    # ── 6. Heatmap Top-12 aeropuertos × año ─────────────────────────────────
    top12 = (
        vuelos["aeropuerto_oaci"].value_counts().head(12).index.tolist()
    )
    pivot = (
        vuelos[vuelos["aeropuerto_oaci"].isin(top12)]
        .assign(anio=vuelos["dt_operacion"].dt.year)
        .groupby(["aeropuerto_oaci", "anio"])
        .size()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(
        pivot / 1_000,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.3,
        fmt=".0f",
        annot=True,
        annot_kws={"fontsize": 6},
        cbar_kws={"label": "Miles de vuelos"},
    )
    ax.set_title("Volumen de vuelos por aeropuerto y año (Top 12)")
    ax.set_xlabel("Año")
    ax.set_ylabel("Aeropuerto OACI")
    fig.tight_layout()
    path = _IMAGES_DIR / "eda_06_heatmap_top12.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append({"archivo": str(path), "descripcion": "Heatmap Top-12 aeropuertos × año"})
    logger.info("Guardado %s", path)

    # ── 7. Correlación entre variables numéricas ─────────────────────────────
    sample = vuelos.sample(min(200_000, len(vuelos)), random_state=_RANDOM_STATE)
    num_cols = ["pmd", "mes_id", "cnt_operaciones", "pmd_fue_imputado"]
    corr = sample[num_cols].assign(
        es_intl=sample["es_internacional"].astype(int)
    ).corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
    )
    ax.set_title("Correlación entre variables numéricas de vuelo")
    fig.tight_layout()
    path = _IMAGES_DIR / "eda_07_correlacion_numericas.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append({"archivo": str(path), "descripcion": "Correlación variables numéricas"})
    logger.info("Guardado %s", path)

    return pd.DataFrame(saved)


# ─────────────────────────────────────────────────────────────────────────────
# CLUSTERING NO SUPERVISADO
# ─────────────────────────────────────────────────────────────────────────────

def cluster_aeropuertos(perfil: pd.DataFrame) -> tuple:
    """K-Means (k=4) sobre el perfil de aeropuertos.

    Features: log(total_vuelos), pct_intl, log(pmd_mediana+1), log(n_aerolineas), log(cnt_ops_media+1)
    Genera ml_01_clusters_aeropuertos.png.
    Devuelve (DataFrame con columna 'cluster', objeto KMeans serializable).
    """
    from sklearn.decomposition import PCA

    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    feat_cols = ["total_vuelos", "pct_intl", "pmd_mediana", "n_aerolineas", "cnt_ops_media"]
    X = perfil[feat_cols].copy()
    X["total_vuelos"] = np.log1p(X["total_vuelos"])
    X["pmd_mediana"]  = np.log1p(X["pmd_mediana"])
    X["n_aerolineas"] = np.log1p(X["n_aerolineas"])
    X["cnt_ops_media"]= np.log1p(X["cnt_ops_media"])

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=4, random_state=_RANDOM_STATE, n_init=20)
    labels = km.fit_predict(Xs)

    pca = PCA(n_components=2, random_state=_RANDOM_STATE)
    coords = pca.fit_transform(Xs)

    resultado = perfil.copy()
    resultado["cluster"] = labels
    resultado["pca_1"] = coords[:, 0]
    resultado["pca_2"] = coords[:, 1]

    CLUSTER_COLORS = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868", 3: "#C44E52"}
    CLUSTER_LABELS = {
        0: "Pequeños / regionales",
        1: "Grandes internacionales",
        2: "Medianos domésticos",
        3: "Alta aviación general",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # PCA scatter
    ax = axes[0]
    for c in range(4):
        mask = resultado["cluster"] == c
        ax.scatter(
            resultado.loc[mask, "pca_1"],
            resultado.loc[mask, "pca_2"],
            label=f"Cluster {c}: {CLUSTER_LABELS[c]}",
            color=CLUSTER_COLORS[c],
            s=80,
            alpha=0.8,
        )
    for _, row in resultado.iterrows():
        ax.annotate(row["aeropuerto_oaci"], (row["pca_1"], row["pca_2"]), fontsize=6, alpha=0.7)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)")
    ax.set_title("Clustering de aeropuertos (K-Means, k=4) — PCA 2D")
    ax.legend(fontsize=8)

    # Volumen vs % internacional scatter
    ax = axes[1]
    for c in range(4):
        mask = resultado["cluster"] == c
        ax.scatter(
            np.log1p(resultado.loc[mask, "total_vuelos"]),
            resultado.loc[mask, "pct_intl"] * 100,
            label=f"Cluster {c}",
            color=CLUSTER_COLORS[c],
            s=80,
            alpha=0.8,
        )
    for _, row in resultado[resultado["total_vuelos"] > 10_000].iterrows():
        ax.annotate(row["aeropuerto_oaci"], (np.log1p(row["total_vuelos"]), row["pct_intl"] * 100), fontsize=6)
    ax.set_xlabel("log(total vuelos)")
    ax.set_ylabel("% vuelos internacionales")
    ax.set_title("Aeropuertos por cluster: volumen vs internacionalidad")
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = _IMAGES_DIR / "ml_01_clusters_aeropuertos.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Guardado %s", path)

    # Tabla resumen por cluster
    resumen = resultado.groupby("cluster").agg(
        n_aeropuertos=("aeropuerto_oaci", "count"),
        total_vuelos_sum=("total_vuelos", "sum"),
        pct_intl_media=("pct_intl", "mean"),
        pmd_mediana_media=("pmd_mediana", "mean"),
    )
    logger.info("Resumen clusters:\n%s", resumen.to_string())
    logger.info("KMeans guardado en catalogo (06_models/kmeans.pkl)")

    return resultado, km


# ─────────────────────────────────────────────────────────────────────────────
# CLASIFICACIÓN SUPERVISADA
# ─────────────────────────────────────────────────────────────────────────────

def train_clasificador(vuelos: pd.DataFrame) -> tuple:
    """RandomForest para predecir `es_internacional`.

    - Muestra estratificada de 500 K filas.
    - Features: aeropuerto_oaci, pmd (log), actividad_cod, tipo_operacion, year, month.
    - class_weight='balanced' para corregir el desbalance 87/13.
    - Genera: ml_02_importancia_features.png, ml_03_confusion_matrix.png.
    - Devuelve (DataFrame metricas, objeto RandomForestClassifier serializable).
    """
    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Muestra estratificada ─────────────────────────────────────────────────
    sample = vuelos.sample(
        n=min(_SAMPLE_SIZE, len(vuelos)),
        random_state=_RANDOM_STATE,
        replace=False,
    )

    # ── Feature engineering ───────────────────────────────────────────────────
    sample = sample.copy()
    sample["year"]  = sample["dt_operacion"].dt.year
    sample["month"] = sample["dt_operacion"].dt.month
    sample["pmd_log"] = np.log1p(sample["pmd"])

    cat_cols = ["aeropuerto_oaci", "actividad_cod", "tipo_operacion"]
    encoders: dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        sample[f"{col}_enc"] = le.fit_transform(sample[col].astype(str))
        encoders[col] = le

    feature_cols = [f"{c}_enc" for c in cat_cols] + ["pmd_log", "year", "month"]
    X = sample[feature_cols].values
    y = sample["es_internacional"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=_RANDOM_STATE, stratify=y
    )

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=_RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # ── Métricas ──────────────────────────────────────────────────────────────
    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)

    logger.info(
        "RF — Accuracy %.3f | Precision %.3f | Recall %.3f | F1 %.3f | ROC-AUC %.3f",
        acc, prec, rec, f1, auc,
    )

    metricas = pd.DataFrame([{
        "modelo": "RandomForestClassifier",
        "target": "es_internacional",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "features": ", ".join(feature_cols),
    }])

    # ── Gráfico: importancia de features ─────────────────────────────────────
    feat_labels = [c.replace("_enc", "") for c in cat_cols] + ["pmd (log)", "año", "mes"]
    importances = pd.Series(rf.feature_importances_, index=feat_labels).sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    importances.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("Importancia (Gini)")
    ax.set_title("Importancia de variables — RandomForest\n(target: es_internacional)")
    fig.tight_layout()
    path = _IMAGES_DIR / "ml_02_importancia_features.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Guardado %s", path)

    # ── Gráfico: matriz de confusión ─────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Doméstico", "Internacional"])
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(
        f"Matriz de confusión — RandomForest\n"
        f"Accuracy {acc:.3f} | F1 {f1:.3f} | ROC-AUC {auc:.3f}"
    )
    fig.tight_layout()
    path = _IMAGES_DIR / "ml_03_confusion_matrix.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Guardado %s", path)
    logger.info("RandomForest guardado en catalogo (06_models/random_forest.pkl)")

    return metricas, rf


# ─────────────────────────────────────────────────────────────────────────────
# EVALUACION FORMAL DEL CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def evaluar_clustering(perfil: pd.DataFrame) -> pd.DataFrame:
    """Curva del codo + silhouette para k=2..8. Justifica k=4.

    Genera ml_04_elbow_silhouette.png.
    Devuelve DataFrame con inertia y silhouette score por k.
    """
    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    feat_cols = ["total_vuelos", "pct_intl", "pmd_mediana", "n_aerolineas", "cnt_ops_media"]
    X = perfil[feat_cols].copy()
    for col in ["total_vuelos", "pmd_mediana", "n_aerolineas", "cnt_ops_media"]:
        X[col] = np.log1p(X[col])
    Xs = StandardScaler().fit_transform(X)

    ks = range(2, 9)
    inertias, silhouettes = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=_RANDOM_STATE, n_init=20)
        labels = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(Xs, labels))
        logger.info("k=%d | inertia=%.1f | silhouette=%.3f", k, km.inertia_, silhouettes[-1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Curva del codo
    ax1.plot(list(ks), inertias, marker="o", linewidth=2, color="#4C72B0")
    ax1.axvline(4, color="#DD8452", linestyle="--", alpha=0.8, label="k=4 elegido")
    ax1.set_xlabel("Numero de clusters (k)")
    ax1.set_ylabel("Inercia (WCSS)")
    ax1.set_title("Curva del Codo — K-Means sobre aeropuertos")
    ax1.legend()

    # Silhouette
    best_k = list(ks)[int(np.argmax(silhouettes))]
    ax2.bar(list(ks), silhouettes, color="#4C72B0", alpha=0.8)
    ax2.bar(4, silhouettes[list(ks).index(4)], color="#DD8452", alpha=0.9, label="k=4 elegido")
    ax2.axhline(silhouettes[list(ks).index(4)], color="#DD8452", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Numero de clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title(f"Silhouette Score por k (maximo en k={best_k})")
    ax2.legend()

    fig.tight_layout()
    path = _IMAGES_DIR / "ml_04_elbow_silhouette.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Guardado %s", path)

    return pd.DataFrame({
        "k": list(ks),
        "inertia": inertias,
        "silhouette": silhouettes,
    })


# ─────────────────────────────────────────────────────────────────────────────
# EVALUACION FORMAL DEL CLASIFICADOR (CV + ROC)
# ─────────────────────────────────────────────────────────────────────────────

def evaluar_clasificador(vuelos: pd.DataFrame) -> pd.DataFrame:
    """Cross-validation 5-fold + curva ROC para el RandomForest.

    Genera ml_05_roc_curve.png.
    Devuelve DataFrame con metricas CV: mean +/- std de accuracy, f1, roc_auc.
    """
    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Muestra estratificada (200 K para que CV sea rapido)
    CV_SAMPLE = 200_000
    sample = vuelos.sample(
        n=min(CV_SAMPLE, len(vuelos)),
        random_state=_RANDOM_STATE,
        replace=False,
    )
    sample = sample.copy()
    sample["year"]    = sample["dt_operacion"].dt.year
    sample["month"]   = sample["dt_operacion"].dt.month
    sample["pmd_log"] = np.log1p(sample["pmd"])

    cat_cols = ["aeropuerto_oaci", "actividad_cod", "tipo_operacion"]
    for col in cat_cols:
        le = LabelEncoder()
        sample[f"{col}_enc"] = le.fit_transform(sample[col].astype(str))

    feature_cols = [f"{c}_enc" for c in cat_cols] + ["pmd_log", "year", "month"]
    X = sample[feature_cols].values
    y = sample["es_internacional"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=_RANDOM_STATE, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        class_weight="balanced",
        random_state=_RANDOM_STATE,
        n_jobs=-1,
    )

    # ── Cross-validation 5-fold ───────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=_RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc",
               "precision": "precision", "recall": "recall"}
    cv_results = cross_validate(rf, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    metricas_cv = pd.DataFrame([{
        "modelo": "RandomForestClassifier",
        "metodo": "StratifiedKFold-5",
        "n_muestra_train": len(X_train),
        "accuracy_mean":   round(cv_results["test_accuracy"].mean(),  4),
        "accuracy_std":    round(cv_results["test_accuracy"].std(),   4),
        "precision_mean":  round(cv_results["test_precision"].mean(), 4),
        "precision_std":   round(cv_results["test_precision"].std(),  4),
        "recall_mean":     round(cv_results["test_recall"].mean(),    4),
        "recall_std":      round(cv_results["test_recall"].std(),     4),
        "f1_mean":         round(cv_results["test_f1"].mean(),        4),
        "f1_std":          round(cv_results["test_f1"].std(),         4),
        "roc_auc_mean":    round(cv_results["test_roc_auc"].mean(),   4),
        "roc_auc_std":     round(cv_results["test_roc_auc"].std(),    4),
    }])
    logger.info(
        "CV-5 — ROC-AUC %.4f +/- %.4f | F1 %.4f +/- %.4f",
        metricas_cv["roc_auc_mean"].iloc[0], metricas_cv["roc_auc_std"].iloc[0],
        metricas_cv["f1_mean"].iloc[0],      metricas_cv["f1_std"].iloc[0],
    )

    # ── Curva ROC (test hold-out) ─────────────────────────────────────────────
    rf.fit(X_train, y_train)
    y_proba = rf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = sklearn_auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#4C72B0", linewidth=2,
            label=f"RandomForest (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="Aleatorio (AUC = 0.5)")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#4C72B0")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR / Recall)")
    ax.set_title("Curva ROC — Clasificador es_internacional\n(RandomForest, muestra 200K)")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    path = _IMAGES_DIR / "ml_05_roc_curve.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Guardado %s", path)

    return metricas_cv
