from __future__ import annotations

import json
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import pytest

from techchallenge.http_benchmark import (
    HttpBenchmarkConfig,
    run_and_log_http_benchmark,
)


class CapturingMlflowClient:
    """Test double that captures only the aggregate MLflow run payload."""

    def __init__(self) -> None:
        self.parameters: dict[str, str] = {}
        self.metrics: dict[str, float] = {}
        self.artifact: dict[str, object] = {}

    def set_tracking_uri(self, uri: str) -> None:
        del uri

    def set_experiment(self, experiment_name: str) -> object:
        assert experiment_name == "kan-24-http-benchmark"
        return object()

    def start_run(self, *, run_name: str) -> AbstractContextManager[object]:
        assert run_name == "sequential_http_benchmark"
        return nullcontext(object())

    def log_params(self, params: dict[str, str]) -> None:
        self.parameters = params

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics = metrics

    def log_artifact(self, local_path: str, artifact_path: str) -> None:
        assert artifact_path == "evaluation"
        self.artifact = json.loads(Path(local_path).read_text(encoding="utf-8"))


class DeterministicBenchmarkClient:
    """Safe fake that counts calls without retaining request text or identifiers."""

    def __init__(self) -> None:
        self.prediction_calls = 0

    def get_model_version(self) -> str:
        return "42"

    def predict(self) -> None:
        self.prediction_calls += 1


def test_http_benchmark_measures_sequential_requests_and_logs_aggregates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = CapturingMlflowClient()
    client = DeterministicBenchmarkClient()
    timestamps = iter((0, 1_000_000, 10, 3_000_010, 20, 2_000_020))
    monkeypatch.setattr(
        "techchallenge.http_benchmark.perf_counter_ns", lambda: next(timestamps)
    )

    result = run_and_log_http_benchmark(
        config=HttpBenchmarkConfig(warmup_requests=2, measured_requests=3),
        client=client,
        tracker=tracker,
    )

    assert client.prediction_calls == 5
    assert result.model_version == "42"
    assert result.mean_latency_ms == pytest.approx(2.0)
    assert result.p50_latency_ms == pytest.approx(2.0)
    assert result.p95_latency_ms == pytest.approx(3.0)
    assert tracker.parameters == {
        "model.version": "42",
        "benchmark_config.api_base_url": "http://localhost:8000",
        "benchmark_config.warmup_requests": "2",
        "benchmark_config.measured_requests": "3",
        "benchmark_config.timeout_seconds": "5.0",
    }
    assert tracker.metrics == {
        "http.mean_latency_ms": pytest.approx(2.0),
        "http.p50_latency_ms": pytest.approx(2.0),
        "http.p95_latency_ms": pytest.approx(3.0),
        "http.measured_requests": 3.0,
    }
    assert tracker.artifact["privacy"] == {
        "contains_record_identifiers": False,
        "contains_request_text": False,
        "contains_response_payloads": False,
    }
    serialized_artifact = json.dumps(tracker.artifact)
    assert "synthetic benchmark input" not in serialized_artifact
    assert "request_id" not in serialized_artifact


@pytest.mark.parametrize(
    "config",
    (
        HttpBenchmarkConfig(warmup_requests=-1),
        HttpBenchmarkConfig(measured_requests=0),
        HttpBenchmarkConfig(timeout_seconds=0.0),
    ),
)
def test_http_benchmark_rejects_invalid_configuration(
    config: HttpBenchmarkConfig,
) -> None:
    with pytest.raises(ValueError):
        run_and_log_http_benchmark(config=config, client=DeterministicBenchmarkClient())
