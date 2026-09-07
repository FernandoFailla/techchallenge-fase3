from __future__ import annotations

import json
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import polars as pl
import pytest

from techchallenge.baseline_nlp import BaselineNlpConfig, BaselineSelection, SplitData
from techchallenge.onnx_benchmark import (
    OnnxBenchmarkConfig,
    _deterministic_references,
    _measure_individual_prediction_latencies,
    _percentile,
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


def test_onnx_benchmark_default_acceptance_protocol() -> None:
    config = OnnxBenchmarkConfig()

    assert config.warmup_predictions == 20
    assert config.measured_predictions == 500


def test_onnx_benchmark_repeats_deterministic_validation_references() -> None:
    references = _deterministic_references(
        SplitData(texts=("first", "second"), targets=("low", "high")), 5
    )

    assert references == ("first", "first", "first", "second", "second")


def test_onnx_benchmark_measures_each_prediction_and_aggregates_percentiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter((0, 1_000_000, 10, 3_000_010, 20, 2_000_020))
    prediction_calls = 0

    def predict_one(_: str) -> str:
        nonlocal prediction_calls
        prediction_calls += 1
        return "label"

    monkeypatch.setattr(
        "techchallenge.onnx_benchmark.perf_counter_ns", lambda: next(timestamps)
    )

    latencies = _measure_individual_prediction_latencies(
        predict_one, ("first", "second", "third")
    )

    assert prediction_calls == 3
    assert latencies == pytest.approx((1.0, 3.0, 2.0))
    assert _percentile(latencies, 0.50) == pytest.approx(2.0)
    assert _percentile(latencies, 0.95) == pytest.approx(3.0)


def test_onnx_benchmark_has_test_class_parity_and_safe_mlflow_payloads(
    tmp_path: Path,
) -> None:
    tracker = CapturingMlflowClient()

    result = run_and_log_onnx_benchmark(
        _modeling_base(),
        baseline_config=_baseline_config(),
        benchmark_config=OnnxBenchmarkConfig(
            warmup_predictions=0,
            measured_predictions=3,
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
    assert result.final_benchmark.warmup_predictions == 0
    assert result.final_benchmark.measured_predictions == 3
    assert result.final_benchmark.sklearn_mean_latency_ms > 0
    assert result.final_benchmark.sklearn_p50_latency_ms > 0
    assert result.final_benchmark.sklearn_p95_latency_ms > 0
    assert result.final_benchmark.onnx_mean_latency_ms > 0
    assert result.final_benchmark.onnx_p50_latency_ms > 0
    assert result.final_benchmark.onnx_p95_latency_ms > 0
    assert [run["name"] for run in tracker.runs] == [
        "validation_selection-dummy_majority",
        "validation_selection-tfidf_random_forest",
        "final_test_report-tfidf_random_forest",
        "onnx_conversion_benchmark",
        "final_test_prediction_parity",
        "final_onnx_benchmark",
    ]
    assert tracker.experiment_names == ["kan-10-onnx-benchmark"] * 6
    final_metrics = tracker.runs[-1]["metrics"]
    assert isinstance(final_metrics, dict)
    assert {
        "benchmark.sklearn_mean_latency_ms",
        "benchmark.sklearn_p50_latency_ms",
        "benchmark.sklearn_p95_latency_ms",
        "benchmark.onnx_mean_latency_ms",
        "benchmark.onnx_p50_latency_ms",
        "benchmark.onnx_p95_latency_ms",
        "benchmark.measured_predictions",
        "benchmark.speedup",
    }.issubset(final_metrics)
    final_artifact = tracker.runs[-1]["artifact"]
    assert isinstance(final_artifact, dict)
    assert final_artifact["benchmark"] == {
        "warmup_predictions": 0,
        "measured_predictions": 3,
        "sklearn_mean_latency_ms": result.final_benchmark.sklearn_mean_latency_ms,
        "sklearn_p50_latency_ms": result.final_benchmark.sklearn_p50_latency_ms,
        "sklearn_p95_latency_ms": result.final_benchmark.sklearn_p95_latency_ms,
        "onnx_mean_latency_ms": result.final_benchmark.onnx_mean_latency_ms,
        "onnx_p50_latency_ms": result.final_benchmark.onnx_p50_latency_ms,
        "onnx_p95_latency_ms": result.final_benchmark.onnx_p95_latency_ms,
        "speedup": result.final_benchmark.speedup,
    }

    for run in tracker.runs:
        artifact = run["artifact"]
        assert isinstance(artifact, dict)
        privacy = artifact["privacy"]
        assert isinstance(privacy, dict)
        assert {
            "contains_model_artifact": False,
            "contains_record_identifiers": False,
            "contains_text": False,
        }.items() <= privacy.items()
        serialized_artifact = json.dumps(artifact)
        assert "critical indicator" not in serialized_artifact
        assert "stable indicator" not in serialized_artifact
        assert "monitor indicator" not in serialized_artifact
    for run in tracker.runs[-3:]:
        artifact = run["artifact"]
        assert isinstance(artifact, dict)
        privacy = artifact["privacy"]
        assert isinstance(privacy, dict)
        assert privacy["contains_predictions"] is False


def test_onnx_benchmark_keeps_unpruned_model_when_fallback_cannot_meet_gate(
    tmp_path: Path,
) -> None:
    tracker = CapturingMlflowClient()

    result = run_and_log_onnx_benchmark(
        _modeling_base(),
        baseline_config=_baseline_config(),
        benchmark_config=OnnxBenchmarkConfig(
            warmup_predictions=0,
            measured_predictions=2,
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
            warmup_predictions=0,
            measured_predictions=3,
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
