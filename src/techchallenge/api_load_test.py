"""Generate safe concurrent traffic for the local API observability dashboard."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from math import ceil
from statistics import fmean
from time import perf_counter_ns
from typing import Final, Protocol
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DEFAULT_API_BASE_URL: Final = "http://localhost:8000"
DEFAULT_REQUEST_COUNT: Final = 500
DEFAULT_CONCURRENCY: Final = 20
DEFAULT_TIMEOUT_SECONDS: Final = 5.0
_SYNTHETIC_TERMS: Final = (
    "amber",
    "birch",
    "cobalt",
    "dawn",
    "ember",
    "fern",
    "glacier",
    "harbor",
)


@dataclass(frozen=True)
class ApiLoadTestConfig:
    """Configuration for synthetic concurrent requests to one API instance."""

    api_base_url: str = DEFAULT_API_BASE_URL
    request_count: int = DEFAULT_REQUEST_COUNT
    concurrency: int = DEFAULT_CONCURRENCY
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class ApiLoadTestResult:
    """Aggregate load outcome without request bodies, responses, or error details."""

    requested: int
    successful: int
    failed: int
    elapsed_seconds: float
    requests_per_second: float
    mean_latency_ms: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None


@dataclass(frozen=True)
class _RequestOutcome:
    """One latency measurement retaining no request or response payload."""

    successful: bool
    latency_ms: float


class ApiLoadTestClient(Protocol):
    """Minimal prediction interface used by the concurrent load generator."""

    def predict(self, text: str) -> None:
        """Submit one synthetic request and discard its response payload."""


class UrllibApiLoadTestClient:
    """Standard-library client that never stores input text or API responses."""

    def __init__(self, api_base_url: str, timeout_seconds: float) -> None:
        parsed_url = urlparse(api_base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("api_base_url must be an absolute HTTP URL")
        self._api_base_url = api_base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def predict(self, text: str) -> None:
        request = Request(
            f"{self._api_base_url}/predict",
            data=json.dumps({"text": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self._timeout_seconds) as response:
            if response.status != 200:
                raise RuntimeError("Load-test prediction request was unsuccessful")


def run_api_load_test(
    *,
    config: ApiLoadTestConfig = ApiLoadTestConfig(),
    client: ApiLoadTestClient | None = None,
) -> ApiLoadTestResult:
    """Send synthetic requests concurrently and return aggregate observability data."""
    _validate_config(config)
    load_client = client or UrllibApiLoadTestClient(
        config.api_base_url, config.timeout_seconds
    )
    started_at = perf_counter_ns()
    outcomes = _run_requests(load_client, config)
    elapsed_seconds = (perf_counter_ns() - started_at) / 1_000_000_000
    successful_latencies = tuple(
        outcome.latency_ms for outcome in outcomes if outcome.successful
    )
    successful = len(successful_latencies)
    return ApiLoadTestResult(
        requested=config.request_count,
        successful=successful,
        failed=config.request_count - successful,
        elapsed_seconds=elapsed_seconds,
        requests_per_second=(config.request_count / elapsed_seconds)
        if elapsed_seconds > 0.0
        else 0.0,
        mean_latency_ms=fmean(successful_latencies) if successful_latencies else None,
        p50_latency_ms=_percentile(successful_latencies, 0.50)
        if successful_latencies
        else None,
        p95_latency_ms=_percentile(successful_latencies, 0.95)
        if successful_latencies
        else None,
    )


def _run_requests(
    client: ApiLoadTestClient, config: ApiLoadTestConfig
) -> tuple[_RequestOutcome, ...]:
    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        futures: tuple[Future[_RequestOutcome], ...] = tuple(
            executor.submit(_measure_request, client, request_index)
            for request_index in range(config.request_count)
        )
        return tuple(future.result() for future in as_completed(futures))


def _measure_request(client: ApiLoadTestClient, request_index: int) -> _RequestOutcome:
    started_at = perf_counter_ns()
    try:
        client.predict(_synthetic_text(request_index))
    except Exception:
        return _RequestOutcome(
            successful=False,
            latency_ms=(perf_counter_ns() - started_at) / 1_000_000,
        )
    return _RequestOutcome(
        successful=True,
        latency_ms=(perf_counter_ns() - started_at) / 1_000_000,
    )


def _synthetic_text(request_index: int) -> str:
    term = _SYNTHETIC_TERMS[request_index % len(_SYNTHETIC_TERMS)]
    return f"Synthetic observability request {term} {request_index}"


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered_values = tuple(sorted(values))
    index = ceil(len(ordered_values) * percentile) - 1
    return ordered_values[index]


def _validate_config(config: ApiLoadTestConfig) -> None:
    parsed_url = urlparse(config.api_base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("api_base_url must be an absolute HTTP URL")
    if config.request_count < 1:
        raise ValueError("request_count must be at least one")
    if config.concurrency < 1:
        raise ValueError("concurrency must be at least one")
    if config.timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive")


def _environment_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    return default if raw_value is None else int(raw_value)


def _environment_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    return default if raw_value is None else float(raw_value)


def main() -> None:
    """Run the default synthetic load test and print only aggregate results."""
    result = run_api_load_test(
        config=ApiLoadTestConfig(
            api_base_url=os.environ.get("API_LOAD_TEST_URL", DEFAULT_API_BASE_URL),
            request_count=_environment_int(
                "API_LOAD_TEST_REQUESTS", DEFAULT_REQUEST_COUNT
            ),
            concurrency=_environment_int(
                "API_LOAD_TEST_CONCURRENCY", DEFAULT_CONCURRENCY
            ),
            timeout_seconds=_environment_float(
                "API_LOAD_TEST_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS
            ),
        )
    )
    print(
        "API load test completed: "
        f"requested={result.requested}, successful={result.successful}, "
        f"failed={result.failed}, "
        f"requests_per_second={result.requests_per_second:.2f}, "
        f"p95_latency_ms={result.p95_latency_ms}"
    )


if __name__ == "__main__":
    main()
