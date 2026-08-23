from __future__ import annotations

import json
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import polars as pl
import pytest

from techchallenge.baseline_nlp import (
    BaselineNlpConfig,
    load_modeling_base,
    read_dvc_provenance,
    run_and_log_experiment,
    split_modeling_base,
)


class CapturingMlflowClient:
    """Test double that records only the payload provided to MLflow."""

    def __init__(self) -> None:
        self.tracking_uri = ""
        self.experiment_name = ""
        self.runs: list[dict[str, object]] = []
        self._active_run: dict[str, object] | None = None

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, experiment_name: str) -> object:
        self.experiment_name = experiment_name
        return object()

    def start_run(self, *, run_name: str) -> AbstractContextManager[object]:
        self._active_run = {"name": run_name, "parameters": {}, "metrics": {}}
        self.runs.append(self._active_run)
        return nullcontext(object())

    def log_params(self, params: dict[str, str]) -> None:
        assert self._active_run is not None
        self._active_run["parameters"] = params

    def log_metrics(self, metrics: dict[str, float]) -> None:
        assert self._active_run is not None
        self._active_run["metrics"] = metrics

    def log_artifact(self, local_path: str, artifact_path: str) -> None:
        assert artifact_path == "evaluation"
        assert self._active_run is not None
        self._active_run["artifact"] = json.loads(
            Path(local_path).read_text(encoding="utf-8")
        )


def _modeling_base() -> pl.DataFrame:
    rows: list[dict[str, str]] = []
    terms = {"high": "critical", "low": "stable", "medium": "monitor"}
    for split, records_per_label in (("train", 3), ("validation", 2), ("test", 2)):
        for label, term in terms.items():
            for index in range(records_per_label):
                rows.append(
                    {
                        "patient_text_en": f"{term} token {split} {index}",
                        "urgency": label,
                        "split": split,
                        "duplicate_cluster": f"{split}-{label}-{index}",
                    }
                )
    return pl.DataFrame(rows)


def _dvc_pointer(path: Path) -> Path:
    pointer_path = path / "modeling_base.parquet.dvc"
    pointer_path.write_text(
        "outs:\n- md5: 0123456789abcdef0123456789abcdef\n"
        "  path: modeling_base.parquet\n",
        encoding="utf-8",
    )
    return pointer_path


def test_load_and_split_modeling_base_uses_only_required_columns(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "modeling_base.parquet"
    _modeling_base().with_columns(pl.lit("ignored").alias("unused")).write_parquet(
        data_path
    )

    splits = split_modeling_base(load_modeling_base(data_path))

    assert splits.train.records == 9
    assert splits.validation.records == 6
    assert splits.test.records == 6
    assert splits.labels == ("high", "low", "medium")


def test_split_modeling_base_rejects_missing_reserved_split() -> None:
    incomplete = _modeling_base().filter(pl.col("split") != "test")

    with pytest.raises(ValueError, match="exactly train, validation, and test"):
        split_modeling_base(incomplete)


def test_read_dvc_provenance_reads_only_valid_output_hash(tmp_path: Path) -> None:
    provenance = read_dvc_provenance(_dvc_pointer(tmp_path))

    assert provenance.pointer_path.endswith("modeling_base.parquet.dvc")
    assert provenance.md5 == "0123456789abcdef0123456789abcdef"


def test_experiment_logs_safe_aggregate_payloads_and_reserves_test(
    tmp_path: Path,
) -> None:
    tracker = CapturingMlflowClient()

    result = run_and_log_experiment(
        _modeling_base(),
        config=BaselineNlpConfig(
            min_document_frequency=1,
            random_forest_estimators=5,
            random_forest_n_jobs=1,
        ),
        dvc_pointer_path=_dvc_pointer(tmp_path),
        experiment_name="baseline-nlp-test",
        tracking_uri="http://mlflow.test:5000",
        tracker=tracker,
    )

    assert tracker.tracking_uri == "http://mlflow.test:5000"
    assert tracker.experiment_name == "baseline-nlp-test"
    assert len(tracker.runs) == 3
    assert [run["name"] for run in tracker.runs] == [
        "validation_selection-dummy_majority",
        "validation_selection-tfidf_random_forest",
        f"final_test_report-{result.selected_model_name}",
    ]
    assert result.final_test_result.split_name == "test"
    assert result.final_test_result.records == 6
    assert set(result.final_test_result.metrics) == {
        "accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
    }
    assert len(result.final_test_result.confusion_matrix) == len(
        result.final_test_result.labels
    )
    assert all(
        len(row) == len(result.final_test_result.labels)
        for row in result.final_test_result.confusion_matrix
    )

    for run in tracker.runs:
        parameters = run["parameters"]
        metrics = run["metrics"]
        artifact = run["artifact"]
        assert isinstance(parameters, dict)
        assert isinstance(metrics, dict)
        assert isinstance(artifact, dict)
        assert parameters["data.dvc_md5"] == "0123456789abcdef0123456789abcdef"
        assert "data.dvc_pointer" in parameters
        assert metrics
        assert artifact["privacy"] == {
            "contains_model_artifact": False,
            "contains_record_identifiers": False,
            "contains_text": False,
        }
        serialized_artifact = json.dumps(artifact)
        assert "critical token" not in serialized_artifact
        assert "stable token" not in serialized_artifact
        assert "monitor token" not in serialized_artifact
