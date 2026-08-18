"""Smoke tests for the reusable package."""

import techchallenge


def test_package_is_importable() -> None:
    """The installed project exposes its reusable package."""
    assert techchallenge.__doc__ is not None
