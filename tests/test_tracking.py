import pytest

from techchallenge.tracking import DEFAULT_TRACKING_URI, get_tracking_uri


def test_tracking_uri_defaults_to_local_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    assert get_tracking_uri() == DEFAULT_TRACKING_URI


def test_tracking_uri_uses_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    assert get_tracking_uri() == "http://mlflow:5000"
