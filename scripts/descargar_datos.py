"""Descarga las tres fuentes del proyecto a data/01_raw/.

Deja el repositorio listo para `kedro run` sin ningun paso manual:

    python scripts/descargar_datos.py

Fuentes:
  - Bitacora de vuelos y operaciones por aeropuerto — datos.gob.cl (JAC)
  - Clima diario de Puerto Montt — Meteostat, estacion 85799 (El Tepual)

Los datos no se versionan (pesan 137 MB y `data/` esta en .gitignore), asi que
este script **es** la garantia de reproducibilidad: cualquiera clona el repo,
lo ejecuta y obtiene exactamente las mismas fuentes.
"""
from __future__ import annotations

import gzip
import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

DESTINO = Path("data/01_raw")

# --- datos.gob.cl — Junta de Aeronautica Civil -------------------------------
JAC = "https://datos.gob.cl/dataset/d5fac3ed-01c7-43f0-9dc4-8970566bc059/resource"
ARCHIVOS_JAC = {
    "bitacora-vuelos.parquet":
        f"{JAC}/5bc8842e-d95a-4f8f-8ec6-acfbe700258a/download/bitacora-vuelos.parquet",
    "operaciones-aeropuertos.csv":
        f"{JAC}/2b7f3fbd-8be8-45e6-8ef5-64eea097528b/download/operaciones-aeropuertos.csv",
}

# --- Meteostat — estacion 85799 (Puerto Montt / El Tepual) -------------------
ESTACION = "85799"
URL_CLIMA = f"https://bulk.meteostat.net/v2/daily/{ESTACION}.csv.gz"
CLIMA_DESDE, CLIMA_HASTA = "2020-01-01", "2026-01-01"
# El bulk de Meteostat viene sin encabezado; este es el orden documentado.
COLUMNAS_CLIMA = ["fecha", "tavg", "tmin", "tmax", "prcp", "snow",
                  "wdir", "wspd", "wpgt", "pres", "tsun"]


def _descargar_jac() -> None:
    for nombre, url in ARCHIVOS_JAC.items():
        destino = DESTINO / nombre
        if destino.exists():
            print(f"  {nombre} ya existe ({destino.stat().st_size / 1e6:.1f} MB), se omite")
            continue
        print(f"  descargando {nombre}...", flush=True)
        urllib.request.urlretrieve(url, destino)
        print(f"  {nombre} — {destino.stat().st_size / 1e6:.1f} MB")


def _descargar_clima() -> None:
    destino = DESTINO / "clima_diario_scte.csv"
    with urllib.request.urlopen(URL_CLIMA, timeout=120) as resp:
        crudo = gzip.decompress(resp.read())

    clima = pd.read_csv(io.BytesIO(crudo), names=COLUMNAS_CLIMA, parse_dates=["fecha"])
    clima = clima[(clima["fecha"] >= CLIMA_DESDE) & (clima["fecha"] < CLIMA_HASTA)]
    clima.to_csv(destino, index=False)

    esperados = (pd.Timestamp(CLIMA_HASTA) - pd.Timestamp(CLIMA_DESDE)).days
    print(f"  clima_diario_scte.csv — {len(clima)} dias de {esperados} esperados "
          f"({clima.fecha.min():%Y-%m-%d} a {clima.fecha.max():%Y-%m-%d})")
    if len(clima) < esperados:
        print(f"  AVISO: faltan {esperados - len(clima)} dias. El bulk de Meteostat "
              f"suele ir ~1 mes atrasado respecto de hoy.")


def main() -> int:
    DESTINO.mkdir(parents=True, exist_ok=True)

    print("datos.gob.cl — Junta de Aeronautica Civil")
    _descargar_jac()
    print("Meteostat — estacion 85799 (Puerto Montt)")
    _descargar_clima()

    print("\nArchivos en data/01_raw:")
    for archivo in sorted(DESTINO.glob("*")):
        if archivo.name != ".gitkeep":
            print(f"  {archivo.name:32s} {archivo.stat().st_size / 1e6:8.1f} MB")
    print("\nListo. Ahora: kedro run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
