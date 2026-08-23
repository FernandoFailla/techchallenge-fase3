"""Shared MLflow tracking configuration and safe aggregate logging."""

from __future__ import annotations

import json
import os
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, TypeAlias, cast

import mlflow

DEFAULT_TRACKING_URI = "http://localhost:5000"

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class MlflowClient(Protocol):
    """Minimal MLflow API used by training code and test doubles."""

    def set_tracking_uri(self, uri: str) -> None: ...

    def set_experiment(self, experiment_name: str) -> object: ...

    def start_run(self, *, run_name: str) -> AbstractContextManager[object]: ...

    def log_params(self, params: dict[str, str]) -> None: ...

    def log_metrics(self, metrics: dict[str, float]) -> None: ...

    def log_artifact(self, local_path: str, artifact_path: str) -> None: ...


@dataclass(frozen=True)
class AggregateTrackingRun:
    """A run payload intentionally restricted to non-sensitive aggregates."""

    run_name: str
    parameters: dict[str, str]
    metrics: dict[str, float]
    artifact: dict[str, JsonValue]


def get_tracking_uri() -> str:
    """Return the local tracking URI or an explicit environment override."""
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def log_aggregate_run(
    tracking_run: AggregateTrackingRun,
    *,
    experiment_name: str,
    tracking_uri: str | None = None,
    client: MlflowClient | None = None,
) -> None:
    """Log one training/evaluation run with a JSON-safe aggregate artifact only."""
    mlflow_client = client if client is not None else cast(MlflowClient, mlflow)
    mlflow_client.set_tracking_uri(tracking_uri or get_tracking_uri())
    mlflow_client.set_experiment(experiment_name)
    with TemporaryDirectory() as temporary_directory:
        artifact_path = Path(temporary_directory) / "evaluation.json"
        artifact_path.write_text(
            json.dumps(tracking_run.artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with mlflow_client.start_run(run_name=tracking_run.run_name):
            mlflow_client.log_params(tracking_run.parameters)
            mlflow_client.log_metrics(tracking_run.metrics)
            mlflow_client.log_artifact(str(artifact_path), artifact_path="evaluation")
