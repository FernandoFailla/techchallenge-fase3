"""Shared MLflow tracking configuration."""

from __future__ import annotations

import os

DEFAULT_TRACKING_URI = "http://localhost:5000"


def get_tracking_uri() -> str:
    """Return the local tracking URI or an explicit environment override."""
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
