# Inventario de datos crudos (data/01_raw)

## Archivos

| ruta_relativa | formato | tamano_mb |
| --- | --- | --- |
| bitacora-vuelos.parquet | parquet | 130.8714 |
| clima_diario_scte.csv | csv | 0.1036 |
| operaciones-aeropuertos.csv | csv | 0.2699 |

## Archivos

- **total_archivos**: 3
- **tamano_total_mb**: 131.2449

## Archivos por formato

- **csv**: 2
- **parquet**: 1

## Muestra

- **archivos_inspeccionados**: 2

## Archivo: clima_diario_scte.csv

- **filas_leidas**: 20
- **columnas**: fecha, tavg, tmin, tmax, prcp, snow, wdir, wspd, wpgt, pres, tsun
- **snow (% nulos)**: 100.0
- **wpgt (% nulos)**: 100.0
- **tsun (% nulos)**: 100.0
- **rango de fecha**: 2020-01-01 a 2020-01-20

## Archivo: operaciones-aeropuertos.csv

- **filas_leidas**: 20
- **columnas**: mes_id, aeropuerto_oaci, internacional_domestico, cnt_operaciones
- **columnas con nulos**: ninguna
- **rango de mes_id (YYYYMM)**: 200001 a 200001
- **aeropuerto_oaci (valores distintos)**: 14
- **internacional_domestico (valores distintos)**: 2
