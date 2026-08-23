"""FastAPI service that serves the MLflow champion model safely."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Final, Protocol, TypeAlias, cast

import mlflow
import pandas as pd  # type: ignore[import-untyped]
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import RequestResponseEndpoint

from techchallenge.model_registry import CHAMPION_ALIAS, MODEL_NAME
from techchallenge.tracking import get_tracking_uri

MAX_TEXT_LENGTH: Final = 10_000
MODEL_OUTPUT_COLUMN: Final = "urgency"
_METRICS_ENDPOINTS: Final = frozenset({"/predict", "/health", "/metrics"})


class PredictionModel(Protocol):
    """The minimal MLflow PyFunc interface required by the API."""

    def predict(self, model_input: pd.DataFrame) -> pd.DataFrame: ...


@dataclass(frozen=True)
class LoadedChampion:
    """An MLflow model and the immutable Registry version being served."""

    model: PredictionModel
    model_version: str


ModelLoader: TypeAlias = Callable[[], LoadedChampion]


class PredictionRequest(BaseModel):
    """One unpersisted triage text submitted for classification."""

    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)

    @field_validator("text")
    @classmethod
    def text_must_contain_non_whitespace(cls, value: str) -> str:
        """Reject whitespace-only submissions without modifying model input."""
        if not value.strip():
            raise ValueError("Text must not be empty")
        return value


class PredictionResponse(BaseModel):
    """The public prediction contract; confidence data is intentionally absent."""

    classification: str
    model_version: str


class HealthResponse(BaseModel):
    """The loaded-model health contract."""

    status: str
    model_version: str


def load_champion() -> LoadedChampion:
    """Resolve and load the champion once, failing startup if MLflow is unavailable."""
    tracking_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    champion = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    model_version = champion.version
    model_uri = f"models:/{MODEL_NAME}/{model_version}"
    model = cast(PredictionModel, mlflow.pyfunc.load_model(model_uri))
    return LoadedChampion(model=model, model_version=model_version)


def create_app(*, model_loader: ModelLoader = load_champion) -> FastAPI:
    """Create the API with an injectable startup loader for contract tests."""
    loaded_champion: dict[str, LoadedChampion] = {}
    registry = CollectorRegistry()
    request_count = Counter(
        "http_requests_total",
        "Total HTTP requests handled by the API.",
        ("endpoint", "method", "status"),
        registry=registry,
    )
    request_latency = Histogram(
        "http_request_duration_seconds",
        "HTTP request latency in seconds.",
        ("endpoint", "method", "status"),
        registry=registry,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        loaded_champion["value"] = model_loader()
        yield

    app = FastAPI(lifespan=lifespan)

    def current_champion() -> LoadedChampion:
        try:
            return loaded_champion["value"]
        except KeyError as error:
            raise RuntimeError("Champion model is not loaded") from error

    @app.middleware("http")
    async def collect_metrics(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        started_at = perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            labels = {
                "endpoint": _endpoint_label(request),
                "method": request.method,
                "status": str(status_code),
            }
            request_count.labels(**labels).inc()
            request_latency.labels(**labels).observe(perf_counter() - started_at)

    @app.exception_handler(RequestValidationError)
    async def invalid_prediction_request(
        _: Request, __: RequestValidationError
    ) -> JSONResponse:
        """Never reflect a submitted text in a validation response."""
        return JSONResponse(status_code=422, content={"detail": "Invalid request"})

    @app.get("/health", response_model=HealthResponse)
    def health(
        champion: LoadedChampion = Depends(current_champion),
    ) -> HealthResponse:
        """Report the Registry version successfully loaded at startup."""
        return HealthResponse(status="ok", model_version=champion.model_version)

    @app.post("/predict", response_model=PredictionResponse)
    def predict(
        payload: PredictionRequest,
        champion: LoadedChampion = Depends(current_champion),
    ) -> PredictionResponse:
        """Classify one text without logging, storing, or returning confidence data."""
        classification = _predict_classification(champion.model, payload.text)
        return PredictionResponse(
            classification=classification,
            model_version=champion.model_version,
        )

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        """Expose only bounded-cardinality request and latency metrics."""
        return Response(
            content=generate_latest(registry),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    return app


def _predict_classification(model: PredictionModel, text: str) -> str:
    predictions = model.predict(pd.DataFrame({"text": [text]}))
    if list(predictions.columns) != [MODEL_OUTPUT_COLUMN] or len(predictions) != 1:
        raise RuntimeError("Champion model returned an invalid prediction payload")
    classification = predictions.iloc[0][MODEL_OUTPUT_COLUMN]
    if not isinstance(classification, str) or not classification:
        raise RuntimeError("Champion model returned an invalid classification")
    return classification


def _endpoint_label(request: Request) -> str:
    route = request.scope.get("route")
    if isinstance(route, APIRoute) and route.path in _METRICS_ENDPOINTS:
        return route.path
    return "unknown"


app = create_app()
