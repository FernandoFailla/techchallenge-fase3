"""Train and safely track the KAN-11 NLP baseline experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, TypeAlias

import polars as pl
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from techchallenge.tracking import (
    AggregateTrackingRun,
    JsonValue,
    MlflowClient,
    log_aggregate_run,
)

REQUIRED_COLUMNS: Final = frozenset(
    {"patient_text_en", "urgency", "split", "duplicate_cluster"}
)
SPLITS: Final = ("train", "validation", "test")
DEFAULT_EXPERIMENT_NAME: Final = "kan-11-baseline-nlp"
TrainedClassifier: TypeAlias = DummyClassifier | Pipeline


@dataclass(frozen=True)
class BaselineNlpConfig:
    """Reproducible parameters for the baseline comparison."""

    random_state: int = 42
    max_features: int = 5_000
    min_document_frequency: int = 2
    ngram_max: int = 2
    random_forest_estimators: int = 300
    random_forest_max_depth: int | None = None
    random_forest_n_jobs: int = -1


@dataclass(frozen=True)
class DataProvenance:
    """The immutable DVC pointer metadata associated with an experiment."""

    pointer_path: str
    md5: str


@dataclass(frozen=True)
class SplitData:
    """One in-memory dataset split; text must not be persisted or logged."""

    texts: tuple[str, ...]
    targets: tuple[str, ...]

    @property
    def records(self) -> int:
        """Return the number of records without exposing their contents."""
        return len(self.targets)


@dataclass(frozen=True)
class ModelingSplits:
    """The fixed train, validation, and held-out test partitions."""

    train: SplitData
    validation: SplitData
    test: SplitData
    labels: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate metrics and confusion matrix for one model/split pair."""

    model_name: str
    split_name: str
    records: int
    metrics: dict[str, float]
    labels: tuple[str, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ExperimentResult:
    """Validation selection results and exactly one final held-out test report."""

    validation_results: tuple[EvaluationResult, EvaluationResult]
    selected_model_name: str
    final_test_result: EvaluationResult


def load_modeling_base(data_path: Path) -> pl.DataFrame:
    """Load only the columns required for training from the DVC-tracked Parquet."""
    lazy_frame = pl.scan_parquet(data_path)
    schema = lazy_frame.collect_schema()
    if REQUIRED_COLUMNS.difference(schema.names()):
        raise ValueError("Modeling base does not have the required training columns")
    if any(schema[column] != pl.String for column in REQUIRED_COLUMNS):
        raise ValueError("Modeling base training columns must be strings")
    modeling_base = lazy_frame.select(sorted(REQUIRED_COLUMNS)).collect()
    if (
        modeling_base.select(
            pl.any_horizontal([pl.col(column).is_null() for column in REQUIRED_COLUMNS])
        )
        .to_series()
        .any()
    ):
        raise ValueError("Modeling base training columns contain null values")
    return modeling_base


def split_modeling_base(modeling_base: pl.DataFrame) -> ModelingSplits:
    """Materialize the precomputed partitions while keeping test data untouched."""
    _validate_modeling_base(modeling_base)
    split_values = set(modeling_base.get_column("split").to_list())
    if split_values != set(SPLITS):
        raise ValueError(
            "Modeling base must contain exactly train, validation, and test"
        )

    def build_split(split_name: str) -> SplitData:
        partition = modeling_base.filter(pl.col("split") == split_name).select(
            "patient_text_en", "urgency"
        )
        if partition.is_empty():
            raise ValueError("Modeling base contains an empty required split")
        texts = tuple(str(value) for value in partition.get_column("patient_text_en"))
        targets = tuple(str(value) for value in partition.get_column("urgency"))
        return SplitData(texts=texts, targets=targets)

    labels = tuple(
        sorted(str(value) for value in modeling_base.get_column("urgency").unique())
    )
    if len(labels) < 2:
        raise ValueError("Modeling base must contain at least two target classes")
    return ModelingSplits(
        train=build_split("train"),
        validation=build_split("validation"),
        test=build_split("test"),
        labels=labels,
    )


def read_dvc_provenance(pointer_path: Path) -> DataProvenance:
    """Read the DVC output MD5 from its pointer without reading source data."""
    for line in pointer_path.read_text(encoding="utf-8").splitlines():
        normalized = line.strip()
        if normalized.startswith("- md5:"):
            md5 = normalized.partition(":")[2].strip()
            if len(md5) == 32 and all(
                character in "0123456789abcdef" for character in md5
            ):
                return DataProvenance(pointer_path=pointer_path.as_posix(), md5=md5)
    raise ValueError("DVC pointer does not contain a valid output MD5")


def train_dummy_classifier(train_data: SplitData) -> DummyClassifier:
    """Fit the majority-class baseline on the training partition only."""
    classifier = DummyClassifier(strategy="most_frequent")
    classifier.fit(train_data.texts, train_data.targets)
    return classifier


def train_tfidf_random_forest(
    train_data: SplitData, config: BaselineNlpConfig
) -> Pipeline:
    """Fit TF-IDF and Random Forest on the supplied training partition only."""
    classifier = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=config.max_features,
                    min_df=config.min_document_frequency,
                    ngram_range=(1, config.ngram_max),
                ),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=config.random_forest_estimators,
                    max_depth=config.random_forest_max_depth,
                    class_weight="balanced",
                    n_jobs=config.random_forest_n_jobs,
                    random_state=config.random_state,
                ),
            ),
        ]
    )
    classifier.fit(train_data.texts, train_data.targets)
    return classifier


def evaluate_classifier(
    classifier: TrainedClassifier,
    data: SplitData,
    *,
    labels: tuple[str, ...],
    model_name: str,
    split_name: str,
) -> EvaluationResult:
    """Calculate aggregate classification metrics without retaining predictions."""
    predictions = tuple(
        str(prediction) for prediction in classifier.predict(data.texts)
    )
    matrix = confusion_matrix(data.targets, predictions, labels=labels)
    metrics = {
        "accuracy": float(accuracy_score(data.targets, predictions)),
        "macro_f1": float(
            f1_score(
                data.targets,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                data.targets,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                data.targets,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
    }
    return EvaluationResult(
        model_name=model_name,
        split_name=split_name,
        records=data.records,
        metrics=metrics,
        labels=labels,
        confusion_matrix=tuple(
            tuple(int(value) for value in row) for row in matrix.tolist()
        ),
    )


def run_and_log_experiment(
    modeling_base: pl.DataFrame,
    *,
    config: BaselineNlpConfig,
    dvc_pointer_path: Path,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: str | None = None,
    tracker: MlflowClient | None = None,
) -> ExperimentResult:
    """Select on validation, then report the selected model once on held-out test."""
    splits = split_modeling_base(modeling_base)
    provenance = read_dvc_provenance(dvc_pointer_path)

    dummy_validation = evaluate_classifier(
        train_dummy_classifier(splits.train),
        splits.validation,
        labels=splits.labels,
        model_name="dummy_majority",
        split_name="validation",
    )
    _log_evaluation(
        dummy_validation,
        config=config,
        provenance=provenance,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        tracker=tracker,
        phase="validation_selection",
    )

    random_forest_validation = evaluate_classifier(
        train_tfidf_random_forest(splits.train, config),
        splits.validation,
        labels=splits.labels,
        model_name="tfidf_random_forest",
        split_name="validation",
    )
    _log_evaluation(
        random_forest_validation,
        config=config,
        provenance=provenance,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        tracker=tracker,
        phase="validation_selection",
    )

    selected_model_name = _select_model(dummy_validation, random_forest_validation)
    train_and_validation = _combine_splits(splits.train, splits.validation)
    selected_classifier: TrainedClassifier
    if selected_model_name == "dummy_majority":
        selected_classifier = train_dummy_classifier(train_and_validation)
    else:
        selected_classifier = train_tfidf_random_forest(train_and_validation, config)
    final_test_result = evaluate_classifier(
        selected_classifier,
        splits.test,
        labels=splits.labels,
        model_name=selected_model_name,
        split_name="test",
    )
    _log_evaluation(
        final_test_result,
        config=config,
        provenance=provenance,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        tracker=tracker,
        phase="final_test_report",
    )
    return ExperimentResult(
        validation_results=(dummy_validation, random_forest_validation),
        selected_model_name=selected_model_name,
        final_test_result=final_test_result,
    )


def _validate_modeling_base(modeling_base: pl.DataFrame) -> None:
    if REQUIRED_COLUMNS.difference(modeling_base.columns):
        raise ValueError("Modeling base does not have the required training columns")
    if any(modeling_base.schema[column] != pl.String for column in REQUIRED_COLUMNS):
        raise ValueError("Modeling base training columns must be strings")
    if (
        modeling_base.select(
            pl.any_horizontal([pl.col(column).is_null() for column in REQUIRED_COLUMNS])
        )
        .to_series()
        .any()
    ):
        raise ValueError("Modeling base training columns contain null values")


def _select_model(
    dummy_result: EvaluationResult, random_forest_result: EvaluationResult
) -> str:
    if random_forest_result.metrics["macro_f1"] > dummy_result.metrics["macro_f1"]:
        return random_forest_result.model_name
    return dummy_result.model_name


def _combine_splits(first: SplitData, second: SplitData) -> SplitData:
    return SplitData(
        texts=first.texts + second.texts,
        targets=first.targets + second.targets,
    )


def _log_evaluation(
    evaluation: EvaluationResult,
    *,
    config: BaselineNlpConfig,
    provenance: DataProvenance,
    experiment_name: str,
    tracking_uri: str | None,
    tracker: MlflowClient | None,
    phase: str,
) -> None:
    parameters = {
        "data.dvc_pointer": provenance.pointer_path,
        "data.dvc_md5": provenance.md5,
        "model.name": evaluation.model_name,
        "run.phase": phase,
        **{f"config.{key}": str(value) for key, value in asdict(config).items()},
    }
    artifact: dict[str, JsonValue] = {
        "data_provenance": {
            "dvc_pointer": provenance.pointer_path,
            "dvc_md5": provenance.md5,
        },
        "evaluation": {
            "confusion_matrix": [list(row) for row in evaluation.confusion_matrix],
            "labels": list(evaluation.labels),
            "metrics": {
                metric_name: metric_value
                for metric_name, metric_value in evaluation.metrics.items()
            },
            "model_name": evaluation.model_name,
            "records": evaluation.records,
            "split": evaluation.split_name,
        },
        "privacy": {
            "contains_model_artifact": False,
            "contains_record_identifiers": False,
            "contains_text": False,
        },
    }
    log_aggregate_run(
        AggregateTrackingRun(
            run_name=f"{phase}-{evaluation.model_name}",
            parameters=parameters,
            metrics={
                f"{evaluation.split_name}.{metric_name}": metric_value
                for metric_name, metric_value in evaluation.metrics.items()
            },
            artifact=artifact,
        ),
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        client=tracker,
    )
