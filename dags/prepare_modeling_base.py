"""On-demand DAG that writes, but never publishes, the modeling base."""

import os
from datetime import datetime
from pathlib import Path

import polars as pl
from airflow.sdk import dag, task

from techchallenge.modeling_base import (
    ModelingBaseConfig,
    assemble_modeling_base,
    assign_grouped_splits,
    build_duplicate_clusters,
    validate_manifest,
    validate_modeling_base_path,
    validate_source_path,
    write_manifest,
    write_parquet,
)

PROJECT_ROOT = Path(os.environ.get("TECHCHALLENGE_ROOT", ".")).resolve()
SOURCE_PATH = PROJECT_ROOT / "data/raw/synthetic_v1.csv"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"


@dag(schedule=None, start_date=datetime(2026, 1, 1), catchup=False, max_active_runs=1)
def prepare_modeling_base() -> None:
    """Materialize a deterministic base for explicit user review."""

    @task
    def validate_source() -> str:
        validate_source_path(SOURCE_PATH)
        return str(SOURCE_PATH)

    @task
    def build_clusters(source_path: str) -> str:
        output_path = PROCESSED_DIR / "duplicate_clusters.parquet"
        write_parquet(
            build_duplicate_clusters(pl.read_csv(source_path), ModelingBaseConfig()),
            output_path,
        )
        return str(output_path)

    @task
    def assign_splits(source_path: str, clusters_path: str) -> str:
        output_path = PROCESSED_DIR / "split_manifest.parquet"
        write_parquet(
            assign_grouped_splits(
                pl.read_csv(source_path),
                pl.read_parquet(clusters_path),
                ModelingBaseConfig(),
            ),
            output_path,
        )
        return str(output_path)

    @task
    def assemble_base(source_path: str, split_path: str) -> str:
        output_path = PROCESSED_DIR / "modeling_base.parquet"
        write_parquet(
            assemble_modeling_base(
                pl.read_csv(source_path), pl.read_parquet(split_path)
            ),
            output_path,
        )
        return str(output_path)

    @task
    def create_manifest(source_path: str, output_path: str) -> str:
        manifest_path = PROCESSED_DIR / "modeling_base.manifest.json"
        write_manifest(
            Path(source_path), Path(output_path), manifest_path, ModelingBaseConfig()
        )
        return str(manifest_path)

    @task
    def validate_artifacts(output_path: str, manifest_path: str) -> None:
        validate_modeling_base_path(Path(output_path))
        validate_manifest(SOURCE_PATH, Path(output_path), Path(manifest_path))

    source_path = validate_source()
    clusters_path = build_clusters(source_path)
    split_path = assign_splits(source_path, clusters_path)
    output_path = assemble_base(source_path, split_path)
    manifest_path = create_manifest(source_path, output_path)
    validate_artifacts(output_path, manifest_path)


prepare_modeling_base()
