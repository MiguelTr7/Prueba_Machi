"""
Inventario de solo lectura de data/01_raw.

Recorre los archivos crudos sin modificarlos y produce un reporte de que hay,
cuanto pesa y con que calidad llega. Es la Fase 2 de CRISP-DM (comprension de
los datos) ejecutada como pipeline, de modo que el diagnostico se puede
reproducir cada vez que cambia una fuente.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

_FORMAT_BY_SUFFIX: dict[str, str] = {
    ".csv": "csv",
    ".parquet": "parquet",
    ".json": "json",
}

# Columnas que, de estar presentes, definen el eje temporal del dataset.
_COLUMNAS_FECHA = ["fecha", "dt_operacion", "fecha_cruce"]

# Columnas categoricas cuya cardinalidad conviene conocer antes de modelar.
_COLUMNAS_CATEGORICAS = [
    "aeropuerto_oaci",
    "aerolinea_dgac",
    "numero_vuelo",
    "modelo_avion",
    "tipo_operacion",
    "actividad_cod",
    "internacional_domestico",
]


def scan_raw_files(raw_data_path: str) -> pd.DataFrame:
    """Index every file under the raw data directory without reading its contents.

    Only filesystem metadata (path, extension, size) is inspected, so this
    is safe to run regardless of how large the raw dataset grows.

    Args:
        raw_data_path: Root directory to scan (e.g. ``data/01_raw``).

    Returns:
        One row per file with its relative path, detected format and size in
        megabytes.
    """
    root = Path(raw_data_path)
    rows: list[dict[str, Any]] = []
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file() or file_path.name == ".gitkeep":
            continue
        suffix = file_path.suffix.lower()
        size_bytes = file_path.stat().st_size
        rows.append(
            {
                "ruta_relativa": file_path.relative_to(root).as_posix(),
                "formato": _FORMAT_BY_SUFFIX.get(suffix, suffix.lstrip(".") or "desconocido"),
                "tamano_mb": round(size_bytes / (1024 * 1024), 4),
            }
        )
    return pd.DataFrame(rows, columns=["ruta_relativa", "formato", "tamano_mb"])


def profile_sample(
    raw_data_path: str, max_files: int, max_rows: int
) -> pd.DataFrame:
    """Profile the schema of a small sample of the raw data, read-only.

    Opens at most ``max_files`` files and, for tabular formats, reads at
    most ``max_rows`` records per file. No values are modified, imputed or
    dropped: class labels, weather strings, etc. are reported exactly as
    they appear in the source, inconsistencies included.

    Args:
        raw_data_path: Root directory holding the raw files.
        max_files: Maximum number of files to open for the sample.
        max_rows: Maximum number of records (rows/frames) to read per file.

    Returns:
        A long-format table with one (categoria, clave, valor) row per
        schema fact discovered: columns present, null rate per column,
        temporal coverage and cardinality of the key categorical columns.
    """
    root = Path(raw_data_path)
    candidate_files = [
        p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() == ".csv"
    ][:max_files]

    facts: list[dict[str, str]] = [
        {"categoria": "muestra", "clave": "archivos_inspeccionados", "valor": str(len(candidate_files))}
    ]
    if not candidate_files:
        facts.append({"categoria": "muestra", "clave": "advertencia", "valor": "sin archivos csv para perfilar"})
        return pd.DataFrame(facts, columns=["categoria", "clave", "valor"])

    # Cada archivo se perfila por separado. Concatenarlos daria un 50% de nulos
    # ficticio en toda columna que solo exista en uno de ellos.
    for archivo in candidate_files:
        nombre = archivo.relative_to(root).as_posix()
        sample = pd.read_csv(archivo, nrows=max_rows)
        categoria = f"archivo: {nombre}"

        facts.append({"categoria": categoria, "clave": "filas_leidas", "valor": str(len(sample))})
        facts.append({"categoria": categoria, "clave": "columnas", "valor": ", ".join(sample.columns)})

        # Calidad: solo se reportan las columnas que efectivamente traen nulos,
        # para que el reporte destaque los problemas en vez de enterrarlos.
        nulos = (sample.isna().mean() * 100).round(2)
        con_nulos = nulos[nulos > 0].sort_values(ascending=False)
        if con_nulos.empty:
            facts.append({"categoria": categoria, "clave": "columnas con nulos", "valor": "ninguna"})
        else:
            for columna, pct in con_nulos.items():
                facts.append({"categoria": categoria, "clave": f"{columna} (% nulos)", "valor": f"{pct}"})

        # Cobertura temporal: define el periodo que el proyecto puede modelar.
        for columna in _COLUMNAS_FECHA:
            if columna not in sample.columns:
                continue
            valores = pd.to_datetime(sample[columna], errors="coerce").dropna()
            if not valores.empty:
                facts.append(
                    {
                        "categoria": categoria,
                        "clave": f"rango de {columna}",
                        "valor": f"{valores.min():%Y-%m-%d} a {valores.max():%Y-%m-%d}",
                    }
                )

        # mes_id viene como entero YYYYMM: se reporta crudo, no como fecha.
        if "mes_id" in sample.columns:
            facts.append(
                {
                    "categoria": categoria,
                    "clave": "rango de mes_id (YYYYMM)",
                    "valor": f"{sample['mes_id'].min()} a {sample['mes_id'].max()}",
                }
            )

        # Cardinalidad: anticipa que variables necesitaran codificacion y
        # cuales tienen demasiados niveles para entrar directo a un modelo.
        for columna in _COLUMNAS_CATEGORICAS:
            if columna in sample.columns:
                facts.append(
                    {
                        "categoria": categoria,
                        "clave": f"{columna} (valores distintos)",
                        "valor": str(sample[columna].nunique()),
                    }
                )

    return pd.DataFrame(facts, columns=["categoria", "clave", "valor"])


def build_inventory_report(
    raw_files_index: pd.DataFrame, raw_sample_profile: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    """Consolidate the file index and the schema profile into one summary.

    Args:
        raw_files_index: Output of :func:`scan_raw_files`.
        raw_sample_profile: Output of :func:`profile_sample`.

    Returns:
        A tuple of (tabular summary combining both inputs, human-readable
        Markdown report).
    """
    file_summary_facts = [
        {"categoria": "archivos", "clave": "total_archivos", "valor": str(len(raw_files_index))},
        {
            "categoria": "archivos",
            "clave": "tamano_total_mb",
            "valor": f"{raw_files_index['tamano_mb'].sum():.4f}" if not raw_files_index.empty else "0",
        },
    ]
    if not raw_files_index.empty:
        for formato, conteo in raw_files_index["formato"].value_counts().items():
            file_summary_facts.append({"categoria": "archivos_por_formato", "clave": formato, "valor": str(conteo)})

    summary = pd.concat(
        [pd.DataFrame(file_summary_facts, columns=["categoria", "clave", "valor"]), raw_sample_profile],
        ignore_index=True,
    )

    report_md = _render_markdown_report(raw_files_index, summary)
    return summary, report_md


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table without extra dependencies."""
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header, separator, *body])


def _render_markdown_report(raw_files_index: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Render the consolidated summary as a readable Markdown document."""
    lines = ["# Inventario de datos crudos (data/01_raw)", ""]

    lines += ["## Archivos", ""]
    lines.append(_df_to_markdown_table(raw_files_index) if not raw_files_index.empty else "_Sin archivos encontrados._")
    lines.append("")

    for categoria, grupo in summary.groupby("categoria", sort=False):
        # Los nombres de archivo se dejan intactos; solo las categorias
        # genericas se formatean para leerse como titulo.
        titulo = categoria if categoria.startswith("archivo: ") else categoria.replace("_", " ")
        lines.append(f"## {titulo[:1].upper()}{titulo[1:]}")
        lines.append("")
        for _, row in grupo.iterrows():
            lines.append(f"- **{row['clave']}**: {row['valor']}")
        lines.append("")

    return "\n".join(lines)
