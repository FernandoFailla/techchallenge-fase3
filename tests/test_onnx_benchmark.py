from __future__ import annotations

import json
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import polars as pl

from techchallenge.baseline_nlp import BaselineNlpConfig, BaselineSelection
from techchallenge.onnx_benchmark import (
    OnnxBenchmarkConfig,
    run_and_log_onnx_benchmark,
)


class CapturingMlflowClient:
    """Test double that keeps only payloads sent to MLflow."""

    def __init__(self) -> None:
        self.tracking_uri = ""
        self.experiment_names: list[str] = []
        self.runs: list[dict[str, object]] = []
        self._active_run: dict[str, object] | None = None

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, experiment_name: str) -> object:
        self.experiment_names.append(experiment_name)
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
    for split, records_per_label in (("train", 5), ("validation", 3), ("test", 3)):
        for label, term in terms.items():
            for index in range(records_per_label):
                rows.append(
                    {
                        "patient_text_en": f"{term} indicator {split} {index}",
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


def _baseline_config() -> BaselineNlpConfig:
    return BaselineNlpConfig(
        min_document_frequency=1,
        random_forest_estimators=10,
        random_forest_n_jobs=1,
    )


def test_onnx_benchmark_has_test_class_parity_and_safe_mlflow_payloads(
    tmp_path: Path,
) -> None:
    tracker = CapturingMlflowClient()

    result = run_and_log_onnx_benchmark(
        _modeling_base(),
        baseline_config=_baseline_config(),
        benchmark_config=OnnxBenchmarkConfig(
            benchmark_records=3,
            warmup_rounds=0,
            repetitions=1,
            desired_speedup=0.0001,
        ),
        dvc_pointer_path=_dvc_pointer(tmp_path),
        tracking_uri="http://mlflow.test:5000",
        tracker=tracker,
    )

    assert result.baseline_result is not None
    assert result.baseline_result.selected_model_name == "tfidf_random_forest"
    assert result.selected_ccp_alpha == 0.0
    assert not result.pruning_attempted
    assert result.onnx_gate_met
    assert result.test_prediction_parity.records == 9
    assert result.test_prediction_parity.parity_rate == 1.0
    assert result.final_benchmark.records == 3
    assert result.final_benchmark.repetitions == 1
    assert result.final_benchmark.sklearn_mean_latency_ms > 0
    assert result.final_benchmark.onnx_mean_latency_ms > 0
    assert [run["name"] for run in tracker.runs] == [
        "validation_selection-dummy_majority",
        "validation_selection-tfidf_random_forest",
        "final_test_report-tfidf_random_forest",
        "onnx_conversion_benchmark",
        "final_test_prediction_parity",
        "final_onnx_benchmark",
    ]
    assert tracker.experiment_names == ["kan-10-onnx-benchmark"] * 6

    for run in tracker.runs:
        artifact = run["artifact"]
        assert isinstance(artifact, dict)
        assert artifact["privacy"] == {
            "contains_model_artifact": False,
            "contains_record_identifiers": False,
            "contains_text": False,
        }
        serialized_artifact = json.dumps(artifact)
        assert "critical indicator" not in serialized_artifact
        assert "stable indicator" not in serialized_artifact
        assert "monitor indicator" not in serialized_artifact


def test_onnx_benchmark_keeps_unpruned_model_when_fallback_cannot_meet_gate(
    tmp_path: Path,
) -> None:
    tracker = CapturingMlflowClient()

    result = run_and_log_onnx_benchmark(
        _modeling_base(),
        baseline_config=_baseline_config(),
        benchmark_config=OnnxBenchmarkConfig(
            benchmark_records=2,
            warmup_rounds=0,
            repetitions=1,
            desired_speedup=1_000_000.0,
            pruning_ccp_alphas=(),
        ),
        dvc_pointer_path=_dvc_pointer(tmp_path),
        tracker=tracker,
    )

    assert result.pruning_attempted
    assert not result.onnx_gate_met
    assert result.selected_ccp_alpha == 0.0
    assert result.test_prediction_parity.parity_rate == 1.0
    assert [run["name"] for run in tracker.runs[-2:]] == [
        "final_test_prediction_parity",
        "final_onnx_benchmark",
    ]


def test_onnx_benchmark_uses_prior_aggregate_selection_without_retraining(
    tmp_path: Path,
) -> None:
    tracker = CapturingMlflowClient()

    result = run_and_log_onnx_benchmark(
        _modeling_base(),
        baseline_config=_baseline_config(),
        benchmark_config=OnnxBenchmarkConfig(
            benchmark_records=3,
            warmup_rounds=0,
            repetitions=1,
            desired_speedup=0.0001,
        ),
        dvc_pointer_path=_dvc_pointer(tmp_path),
        tracker=tracker,
        prior_selection=BaselineSelection(
            selected_model_name="tfidf_random_forest",
            random_forest_validation_macro_f1=1.0,
        ),
    )

    assert result.baseline_result is None
    assert [run["name"] for run in tracker.runs] == [
        "onnx_conversion_benchmark",
        "final_test_prediction_parity",
        "final_onnx_benchmark",
    ]
