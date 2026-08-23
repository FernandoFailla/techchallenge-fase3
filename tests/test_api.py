from __future__ import annotations

from collections.abc import Callable

import pandas as pd  # type: ignore[import-untyped]
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from techchallenge.api import (
    DEFAULT_MODEL_URI,
    MAX_TEXT_LENGTH,
    LoadedChampion,
    _parse_model_uri,
    create_app,
    get_model_uri,
)


class FakeModel:
    """Deterministic model double that does not retain submitted text."""

    def predict(self, model_input: pd.DataFrame) -> pd.DataFrame:
        del model_input
        return pd.DataFrame({"urgency": ["high"]})


def _app_with_loader(
    loader: Callable[[], LoadedChampion] | None = None,
) -> FastAPI:
    return create_app(
        model_loader=(
            loader
            if loader is not None
            else lambda: LoadedChampion(model=FakeModel(), model_version="7")
        )
    )


def test_contract_serves_prediction_health_and_metrics_with_one_startup_load() -> None:
    load_count = 0

    def load_model() -> LoadedChampion:
        nonlocal load_count
        load_count += 1
        return LoadedChampion(model=FakeModel(), model_version="7")

    with TestClient(_app_with_loader(load_model)) as client:
        prediction = client.post("/predict", json={"text": "example input"})
        health = client.get("/health")
        metrics = client.get("/metrics")

    assert prediction.status_code == 200
    assert prediction.json() == {"classification": "high", "model_version": "7"}
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "model_version": "7"}
    assert metrics.status_code == 200
    assert metrics.headers["content-type"].startswith("text/plain")
    assert (
        'http_requests_total{endpoint="/predict",method="POST",status="200"} 1.0'
        in metrics.text
    )
    assert (
        "http_request_duration_seconds_count{"
        'endpoint="/predict",method="POST",status="200"} 1.0' in metrics.text
    )
    assert load_count == 1


@pytest.mark.parametrize("text", ("", "   "))
def test_contract_rejects_empty_text_without_reflecting_input(text: str) -> None:
    with TestClient(_app_with_loader()) as client:
        response = client.post("/predict", json={"text": text})

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request"}


def test_contract_rejects_text_over_maximum_length_without_reflecting_input() -> None:
    text = "x" * (MAX_TEXT_LENGTH + 1)

    with TestClient(_app_with_loader()) as client:
        response = client.post("/predict", json={"text": text})

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request"}
    assert text not in response.text


def test_metrics_do_not_use_submitted_text_or_error_identifiers_as_labels() -> None:
    submitted_text = "canary-input-should-not-appear-in-metrics"

    with TestClient(_app_with_loader()) as client:
        client.post("/predict", json={"text": submitted_text})
        metrics = client.get("/metrics")

    assert submitted_text not in metrics.text
    assert "error_id" not in metrics.text
    assert 'endpoint="/predict"' in metrics.text
    assert 'method="POST"' in metrics.text
    assert 'status="200"' in metrics.text


def test_startup_fails_when_champion_cannot_be_loaded() -> None:
    def unavailable_loader() -> LoadedChampion:
        raise RuntimeError("MLflow champion is unavailable")

    with pytest.raises(RuntimeError, match="MLflow champion is unavailable"):
        with TestClient(_app_with_loader(unavailable_loader)):
            pass


def test_model_uri_defaults_to_champion_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_MODEL_URI", raising=False)

    assert get_model_uri() == DEFAULT_MODEL_URI
    assert _parse_model_uri(get_model_uri()) == (
        "triage-urgency-classifier",
        "@champion",
    )


def test_model_uri_accepts_an_explicit_registry_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_MODEL_URI", "models:/triage-urgency-classifier/12")

    assert _parse_model_uri(get_model_uri()) == ("triage-urgency-classifier", "12")


@pytest.mark.parametrize(
    "model_uri",
    (
        "runs:/123/model",
        "models:/triage-urgency-classifier",
        "models:/triage-urgency-classifier/one",
    ),
)
def test_model_uri_rejects_non_deterministic_references(model_uri: str) -> None:
    with pytest.raises(ValueError, match="MLFLOW_MODEL_URI"):
        _parse_model_uri(model_uri)
