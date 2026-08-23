"""Benchmark sequential HTTP latency of the local champion API safely."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import ceil
from statistics import fmean
from time import perf_counter_ns
from typing import Final, Protocol
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from techchallenge.tracking import (
    AggregateTrackingRun,
    MlflowClient,
    log_aggregate_run,
)

DEFAULT_EXPERIMENT_NAME_HTTP: Final = "kan-24-http-benchmark"
DEFAULT_API_BASE_URL: Final = "http://localhost:8000"
_BENCHMARK_REQUEST_BODY: Final = b'{"text":"synthetic benchmark input"}'


@dataclass(frozen=True)
class HttpBenchmarkConfig:
    """Settings for a sequential benchmark of the local API container."""

    api_base_url: str = DEFAULT_API_BASE_URL
    warmup_requests: int = 20
    measured_requests: int = 200
    timeout_seconds: float = 5.0


@dataclass(frozen=True)
class HttpLatencyBenchmark:
    """Aggregate latency measurements from successful sequential HTTP requests."""

    model_version: str
    warmup_requests: int
    measured_requests: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float


class HttpBenchmarkClient(Protocol):
    """Minimal API contract that avoids retaining benchmark request payloads."""

    def get_model_version(self) -> str:
        """Return the version reported by the ready API health endpoint."""

    def predict(self) -> None:
        """Make one successful prediction request without exposing its response."""


class UrllibHttpBenchmarkClient:
    """HTTP client that sends one fixed synthetic request and retains no responses."""

    def __init__(self, api_base_url: str, timeout_seconds: float) -> None:
        parsed_url = urlparse(api_base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("api_base_url must be an absolute HTTP URL")
        self._api_base_url = api_base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def get_model_version(self) -> str:
        """Read only the non-sensitive health payload needed for experiment metadata."""
        request = Request(f"{self._api_base_url}/health", method="GET")
        with urlopen(request, timeout=self._timeout_seconds) as response:
            if response.status != 200:
                raise RuntimeError(
                    "Benchmark health endpoint returned a non-success status"
                )
            payload = json.loads(response.read())
        if not isinstance(payload, dict):
            raise RuntimeError("Benchmark health endpoint returned an invalid payload")
        status = payload.get("status")
        model_version = payload.get("model_version")
        if status != "ok" or not isinstance(model_version, str) or not model_version:
            raise RuntimeError("Benchmark health endpoint returned an invalid payload")
        return model_version

    def predict(self) -> None:
        """Send the deterministic synthetic request and discard the response body."""
        request = Request(
            f"{self._api_base_url}/predict",
            data=_BENCHMARK_REQUEST_BODY,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self._timeout_seconds) as response:
            if response.status != 200:
                raise RuntimeError(
                    "Benchmark prediction endpoint returned a non-success status"
                )


def run_and_log_http_benchmark(
    *,
    config: HttpBenchmarkConfig = HttpBenchmarkConfig(),
    experiment_name: str = DEFAULT_EXPERIMENT_NAME_HTTP,
    tracking_uri: str | None = None,
    tracker: MlflowClient | None = None,
    client: HttpBenchmarkClient | None = None,
) -> HttpLatencyBenchmark:
    """Measure sequential API calls and log only aggregate measurements to MLflow."""
    _validate_benchmark_config(config)
    benchmark_client = client or UrllibHttpBenchmarkClient(
        config.api_base_url, config.timeout_seconds
    )
    model_version = benchmark_client.get_model_version()
    for _ in range(config.warmup_requests):
        benchmark_client.predict()

    latencies_ms = tuple(
        _measure_request_latency_ms(benchmark_client)
        for _ in range(config.measured_requests)
    )
    result = HttpLatencyBenchmark(
        model_version=model_version,
        warmup_requests=config.warmup_requests,
        measured_requests=config.measured_requests,
        mean_latency_ms=fmean(latencies_ms),
        p50_latency_ms=_percentile(latencies_ms, 0.50),
        p95_latency_ms=_percentile(latencies_ms, 0.95),
    )
    log_aggregate_run(
        AggregateTrackingRun(
            run_name="sequential_http_benchmark",
            parameters={
                "model.version": result.model_version,
                **{
                    f"benchmark_config.{key}": str(value)
                    for key, value in asdict(config).items()
                },
            },
            metrics={
                "http.mean_latency_ms": result.mean_latency_ms,
                "http.p50_latency_ms": result.p50_latency_ms,
                "http.p95_latency_ms": result.p95_latency_ms,
                "http.measured_requests": float(result.measured_requests),
            },
            artifact={
                "benchmark": {
                    "model_version": result.model_version,
                    "warmup_requests": result.warmup_requests,
                    "measured_requests": result.measured_requests,
                    "mean_latency_ms": result.mean_latency_ms,
                    "p50_latency_ms": result.p50_latency_ms,
                    "p95_latency_ms": result.p95_latency_ms,
                },
                "privacy": {
                    "contains_record_identifiers": False,
                    "contains_request_text": False,
                    "contains_response_payloads": False,
                },
            },
        ),
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        client=tracker,
    )
    return result


def _measure_request_latency_ms(client: HttpBenchmarkClient) -> float:
    started_at = perf_counter_ns()
    client.predict()
    return (perf_counter_ns() - started_at) / 1_000_000


def _percentile(values: tuple[float, ...], percentile: float) -> float:
    index = ceil(len(values) * percentile) - 1
    return sorted(values)[index]


def _validate_benchmark_config(config: HttpBenchmarkConfig) -> None:
    if config.warmup_requests < 0:
        raise ValueError("warmup_requests cannot be negative")
    if config.measured_requests < 1:
        raise ValueError("measured_requests must be at least one")
    if config.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")


def main() -> None:
    """Run the default local benchmark and print its safe aggregate outcome."""
    result = run_and_log_http_benchmark()
    print(
        "HTTP benchmark completed: "
        f"model_version={result.model_version}, "
        f"mean_latency_ms={result.mean_latency_ms:.3f}, "
        f"p95_latency_ms={result.p95_latency_ms:.3f}"
    )


if __name__ == "__main__":
    main()
