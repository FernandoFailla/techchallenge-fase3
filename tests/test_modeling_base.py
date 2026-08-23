from pathlib import Path

import polars as pl

from techchallenge.modeling_base import (
    ModelingBaseConfig,
    assemble_modeling_base,
    assign_grouped_splits,
    build_duplicate_clusters,
    validate_manifest,
    write_manifest,
    write_parquet,
)


def _dataset() -> pl.DataFrame:
    rows = [
        {
            "id": f"{label}-{index}",
            "patient_text_en": f"{label} record {index}",
            "urgency": label,
        }
        for label in ("low", "medium", "high")
        for index in range(25)
    ]
    return pl.DataFrame(rows)


def test_modeling_base_is_deterministic_and_keeps_clusters_together() -> None:
    config = ModelingBaseConfig()
    dataset = _dataset()
    clusters = build_duplicate_clusters(dataset, config)
    first = assemble_modeling_base(
        dataset, assign_grouped_splits(dataset, clusters, config)
    )
    second = assemble_modeling_base(
        dataset, assign_grouped_splits(dataset, clusters, config)
    )

    assert first.equals(second)
    assert (
        first.group_by("duplicate_cluster")
        .agg(pl.col("split").n_unique())
        .get_column("split")
        .max()
        == 1
    )


def test_manifest_does_not_contain_source_text(tmp_path: Path) -> None:
    source_path = tmp_path / "source.csv"
    output_path = tmp_path / "base.parquet"
    manifest_path = tmp_path / "base.manifest.json"
    _dataset().write_csv(source_path)

    dataset = _dataset()
    clusters = build_duplicate_clusters(dataset, ModelingBaseConfig())
    modeling_base = assemble_modeling_base(
        dataset, assign_grouped_splits(dataset, clusters, ModelingBaseConfig())
    )
    write_parquet(modeling_base, output_path)
    write_manifest(source_path, output_path, manifest_path, ModelingBaseConfig())
    validate_manifest(source_path, output_path, manifest_path)

    assert output_path.is_file()
    assert "low record 0" not in manifest_path.read_text(encoding="utf-8")
