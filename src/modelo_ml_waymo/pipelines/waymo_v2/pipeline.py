"""
Pipeline 'waymo_v2'
"""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_spark_feature_table,
    build_waymo_v2_report,
    extract_grayscale_images,
    load_and_merge_parquets,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_and_merge_parquets,
                inputs=[
                    "params:lidar_box_dir",
                    "params:assoc_dir",
                    "params:stats_dir",
                ],
                outputs="waymo_v2_features_raw",
                name="load_and_merge_parquets_node",
            ),
            node(
                func=extract_grayscale_images,
                inputs=[
                    "params:camera_image_dir",
                    "params:image_output_dir",
                    "params:stats_dir",
                ],
                outputs="waymo_v2_image_metadata",
                name="extract_grayscale_images_node",
            ),
            node(
                func=build_spark_feature_table,
                inputs=["waymo_v2_features_raw", "params:spark_master"],
                outputs="waymo_v2_features",
                name="build_spark_feature_table_node",
            ),
            node(
                func=build_waymo_v2_report,
                inputs=["waymo_v2_features", "waymo_v2_image_metadata"],
                outputs="waymo_v2_report_md",
                name="build_waymo_v2_report_node",
            ),
        ]
    )
