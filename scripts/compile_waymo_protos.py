"""One-time compiler for the Waymo Open Dataset .proto sources into _pb2.py modules.

The official ``waymo-open-dataset-tf-2-12-0`` pip package has no Windows wheel,
so instead of depending on it we vendor only the .proto files we actually need
(dataset/label/map/vector/keypoint -- enough for the Frame/CameraImage/CameraName
classes used by the image_ingestion pipeline) and compile them ourselves with
grpc_tools.protoc, which is pure Python/protobuf and installs fine on Windows.

The generated ``dataset_pb2.py`` uses protoc's default absolute imports (e.g.
``from waymo_open_dataset import label_pb2``), so ``waymo_open_dataset`` must be
a top-level package directly on ``sys.path`` -- not nested under
``modelo_ml_waymo``. Kedro already puts ``src/`` on ``sys.path``, so we generate
straight into ``src/waymo_open_dataset/`` (a sibling of ``src/modelo_ml_waymo/``)
instead of inside the ``modelo_ml_waymo`` package.

Run once after ``pip install -r requirements.txt`` (or ``uv sync``), before
``kedro run --pipeline=image_ingestion``:

    python scripts/compile_waymo_protos.py

Regenerates the _pb2.py modules under src/waymo_open_dataset/ from the .proto
sources checked into src/modelo_ml_waymo/waymo_protos_src/. Safe to re-run --
it only overwrites the generated files.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTO_SRC_ROOT = REPO_ROOT / "src" / "modelo_ml_waymo" / "waymo_protos_src"
PROTO_OUT_ROOT = REPO_ROOT / "src"

PROTO_FILES = [
    PROTO_SRC_ROOT / "waymo_open_dataset" / "dataset.proto",
    PROTO_SRC_ROOT / "waymo_open_dataset" / "label.proto",
    PROTO_SRC_ROOT / "waymo_open_dataset" / "protos" / "map.proto",
    PROTO_SRC_ROOT / "waymo_open_dataset" / "protos" / "vector.proto",
    PROTO_SRC_ROOT / "waymo_open_dataset" / "protos" / "keypoint.proto",
]


def main() -> None:
    missing = [p for p in PROTO_FILES if not p.is_file()]
    if missing:
        raise SystemExit(
            "Faltan estos .proto en waymo_protos_src (ver README, seccion image_ingestion):\n"
            + "\n".join(f"  - {p}" for p in missing)
        )

    command = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"-I={PROTO_SRC_ROOT}",
        f"--python_out={PROTO_OUT_ROOT}",
        *(str(p) for p in PROTO_FILES),
    ]
    print("Ejecutando:", " ".join(command))
    subprocess.run(command, check=True, cwd=REPO_ROOT)
    print(f"Listo. Modulos _pb2.py generados en {PROTO_OUT_ROOT}")


if __name__ == "__main__":
    main()
