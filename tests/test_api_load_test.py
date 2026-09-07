from __future__ import annotations

import pytest

from techchallenge.api_load_test import ApiLoadTestConfig, run_api_load_test


class RecordingLoadTestClient:
    """Thread-safe-enough test double that retains only synthetic request text."""

    def __init__(self, failing_indexes: frozenset[int] = frozenset()) -> None:
        self.failing_indexes = failing_indexes
        self.requested_texts: list[str] = []

    def predict(self, text: str) -> None:
        self.requested_texts.append(text)
        if int(text.rsplit(" ", maxsplit=1)[1]) in self.failing_indexes:
            raise RuntimeError("Synthetic request failed")


def test_api_load_test_uses_distinct_synthetic_texts_and_reports_aggregates() -> None:
    client = RecordingLoadTestClient(failing_indexes=frozenset({2, 4}))

    result = run_api_load_test(
        config=ApiLoadTestConfig(request_count=6, concurrency=3), client=client
    )

    assert result.requested == 6
    assert result.successful == 4
    assert result.failed == 2
    assert result.requests_per_second > 0.0
    assert result.mean_latency_ms is not None
    assert result.p50_latency_ms is not None
    assert result.p95_latency_ms is not None
    assert len(client.requested_texts) == 6
    assert len(set(client.requested_texts)) == 6
    assert all(
        text.startswith("Synthetic observability request ")
        for text in client.requested_texts
    )


def test_api_load_test_returns_no_latency_percentiles_when_all_requests_fail() -> None:
    client = RecordingLoadTestClient(failing_indexes=frozenset({0, 1}))

    result = run_api_load_test(
        config=ApiLoadTestConfig(request_count=2, concurrency=1), client=client
    )

    assert result.successful == 0
    assert result.failed == 2
    assert result.mean_latency_ms is None
    assert result.p50_latency_ms is None
    assert result.p95_latency_ms is None


@pytest.mark.parametrize(
    "config",
    (
        ApiLoadTestConfig(api_base_url="not-a-url"),
        ApiLoadTestConfig(request_count=0),
        ApiLoadTestConfig(concurrency=0),
        ApiLoadTestConfig(timeout_seconds=0.0),
    ),
)
def test_api_load_test_rejects_invalid_configuration(config: ApiLoadTestConfig) -> None:
    with pytest.raises(ValueError):
        run_api_load_test(config=config, client=RecordingLoadTestClient())
