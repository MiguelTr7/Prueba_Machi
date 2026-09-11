"""
This is a boilerplate pipeline 'image_ingestion'
generated using Kedro 1.3.1
"""
from __future__ import annotations

from kedro.pipeline import Node, Pipeline

from .nodes import build_image_metadata_table, extract_and_grayscale, list_perception_tfrecords


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=list_perception_tfrecords,
                inputs="params:image_ingestion.raw_tfrecord_dir",
                outputs="image_ingestion_tfrecord_index",
                name="list_perception_tfrecords",
            ),
            Node(
                func=extract_and_grayscale,
                inputs=[
                    "image_ingestion_tfrecord_index",
                    "params:image_ingestion.raw_tfrecord_dir",
                    "params:image_ingestion.output_dir",
                ],
                outputs="image_ingestion_records",
                name="extract_and_grayscale",
            ),
            Node(
                func=build_image_metadata_table,
                inputs="image_ingestion_records",
                outputs="image_metadata",
                name="build_image_metadata_table",
            ),
        ]
    )
