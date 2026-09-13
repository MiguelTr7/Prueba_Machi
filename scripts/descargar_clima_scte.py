"""Descarga el clima diario historico de Puerto Montt (El Tepual) desde Meteostat.

Estacion 85799 — la misma que expone https://meteostat.net/es/place/cl/puerto-montt
Genera data/01_raw/clima_diario_scte.csv acotado al periodo del proyecto (2020-2026).

Uso:
    python scripts/descargar_clima_scte.py
"""
from __future__ import annotations

import gzip
import io
import urllib.request
from pathlib import Path

import pandas as pd

ESTACION = "85799"
URL = f"https://bulk.meteostat.net/v2/daily/{ESTACION}.csv.gz"
DESDE, HASTA = "2020-01-01", "2026-01-01"
DESTINO = Path("data/01_raw/clima_diario_scte.csv")

# El bulk de Meteostat viene sin encabezado; este es el orden documentado.
COLUMNAS = ["fecha", "tavg", "tmin", "tmax", "prcp", "snow",
            "wdir", "wspd", "wpgt", "pres", "tsun"]


def main() -> None:
    with urllib.request.urlopen(URL, timeout=120) as resp:
        crudo = gzip.decompress(resp.read())

    clima = pd.read_csv(io.BytesIO(crudo), names=COLUMNAS, parse_dates=["fecha"])
    clima = clima[(clima["fecha"] >= DESDE) & (clima["fecha"] < HASTA)]

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    clima.to_csv(DESTINO, index=False)

    print(f"{DESTINO} — {len(clima)} dias ({clima.fecha.min().date()} a {clima.fecha.max().date()})")
    cobertura = (clima.notna().mean() * 100).round(1)
    print("Cobertura por columna (%):")
    print(cobertura.to_string())


if __name__ == "__main__":
    main()
