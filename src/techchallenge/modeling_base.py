"""Build deterministic, leakage-aware modeling datasets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors

REQUIRED_COLUMNS: Final = {"id", "patient_text_en", "urgency"}


@dataclass(frozen=True)
class ModelingBaseConfig:
    """Configuration that defines a reproducible modeling base."""

    similarity_threshold: float = 0.90
    seed: int = 42
    n_splits: int = 20
    train_folds: int = 14
    validation_folds: int = 3


def build_duplicate_clusters(
    dataset: pl.DataFrame, config: ModelingBaseConfig
) -> pl.DataFrame:
    """Return a cluster for every record using all pairs above the threshold."""
    _validate_source(dataset)
    texts = dataset.get_column("patient_text_en").to_list()
    vectors = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2
    ).fit_transform(texts)
    distances, indices = (
        NearestNeighbors(
            radius=1 - config.similarity_threshold, metric="cosine", n_jobs=-1
        )
        .fit(vectors)
        .radius_neighbors(vectors, return_distance=True)
    )
    parents = list(range(dataset.height))

    def find_root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for source_index, (source_distances, source_neighbors) in enumerate(
        zip(distances, indices, strict=True)
    ):
        for distance, neighbor_index in zip(
            source_distances, source_neighbors, strict=True
        ):
            if (
                source_index >= neighbor_index
                or 1 - distance < config.similarity_threshold
            ):
                continue
            source_root = find_root(source_index)
            neighbor_root = find_root(neighbor_index)
            if source_root != neighbor_root:
                parents[neighbor_root] = source_root

    return pl.DataFrame(
        {
            "row_index": pl.Series(range(dataset.height), dtype=pl.UInt32),
            "duplicate_cluster": [
                f"cluster-{find_root(index)}" for index in range(dataset.height)
            ],
        }
    )


def assign_grouped_splits(
    dataset: pl.DataFrame, clusters: pl.DataFrame, config: ModelingBaseConfig
) -> pl.DataFrame:
    """Assign each duplicate cluster to one deterministic split."""
    _validate_source(dataset)
    groups = clusters.get_column("duplicate_cluster").to_list()
    labels = dataset.get_column("urgency").to_list()
    fold_by_row = [0] * dataset.height
    splitter = StratifiedGroupKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.seed
    )
    for fold, (_, fold_rows) in enumerate(
        splitter.split(range(dataset.height), labels, groups)
    ):
        for row_index in fold_rows:
            fold_by_row[int(row_index)] = fold

    split_manifest = (
        dataset.with_row_index("row_index")
        .select("row_index", "id", "urgency")
        .join(clusters, on="row_index")
        .with_columns(pl.Series("fold", fold_by_row))
        .with_columns(
            pl.when(pl.col("fold") < config.train_folds)
            .then(pl.lit("train"))
            .when(pl.col("fold") < config.train_folds + config.validation_folds)
            .then(pl.lit("validation"))
            .otherwise(pl.lit("test"))
            .alias("split")
        )
        .select("id", "duplicate_cluster", "split")
    )
    _validate_split_manifest(split_manifest)
    return split_manifest


def assemble_modeling_base(
    dataset: pl.DataFrame, split_manifest: pl.DataFrame
) -> pl.DataFrame:
    """Join validated split assignments to the model input columns."""
    _validate_source(dataset)
    modeling_base = dataset.select("id", "patient_text_en", "urgency").join(
        split_manifest, on="id", validate="1:1"
    )
    _validate_modeling_base(modeling_base)
    return modeling_base


def write_parquet(dataframe: pl.DataFrame, output_path: Path) -> None:
    """Write one already-materialized artifact without publishing it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    dataframe.write_parquet(temporary_path)
    temporary_path.replace(output_path)


def write_manifest(
    source_path: Path,
    output_path: Path,
    manifest_path: Path,
    config: ModelingBaseConfig,
) -> None:
    """Write metadata for an existing modeling base without publishing it."""
    modeling_base = pl.read_parquet(output_path)
    temporary_path = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
    temporary_path.write_text(
        _manifest_json(source_path, output_path, modeling_base, config),
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)


def validate_source_path(source_path: Path) -> int:
    """Validate a source file and return only its record count."""
    dataset = pl.read_csv(source_path)
    _validate_source(dataset)
    return dataset.height


def validate_modeling_base_path(modeling_base_path: Path) -> int:
    """Validate a materialized base and return only its record count."""
    modeling_base = pl.read_parquet(modeling_base_path)
    _validate_modeling_base(modeling_base)
    return modeling_base.height


def validate_manifest(
    source_path: Path, modeling_base_path: Path, manifest_path: Path
) -> None:
    """Verify that a metadata-only manifest describes the current artifacts."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["source_sha256"] != _sha256(source_path):
        raise ValueError("Manifest source checksum does not match")
    if manifest["output_sha256"] != _sha256(modeling_base_path):
        raise ValueError("Manifest output checksum does not match")
    if manifest["records"] != validate_modeling_base_path(modeling_base_path):
        raise ValueError("Manifest record count does not match")


def _validate_source(dataset: pl.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(dataset.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if (
        dataset.select(
            pl.any_horizontal([pl.col(name).is_null() for name in REQUIRED_COLUMNS])
        )
        .to_series()
        .any()
    ):
        raise ValueError("Required modeling columns contain null values")


def _validate_modeling_base(modeling_base: pl.DataFrame) -> None:
    _validate_split_manifest(modeling_base)


def _validate_split_manifest(split_manifest: pl.DataFrame) -> None:
    crossing_clusters = (
        split_manifest.group_by("duplicate_cluster")
        .agg(pl.col("split").n_unique().alias("split_count"))
        .filter(pl.col("split_count") > 1)
    )
    if crossing_clusters.height:
        raise ValueError("A duplicate cluster crossed a data split")


def _manifest_json(
    source_path: Path,
    output_path: Path,
    modeling_base: pl.DataFrame,
    config: ModelingBaseConfig,
) -> str:
    distribution = (
        modeling_base.group_by(["split", "urgency"]).len(name="records").to_dicts()
    )
    manifest = {
        "source_sha256": _sha256(source_path),
        "output_sha256": _sha256(output_path),
        "config": asdict(config),
        "records": modeling_base.height,
        "split_distribution": distribution,
    }
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
