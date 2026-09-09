"""Make an absent native extension a CI failure when explicitly required."""

import importlib.util
import os

import pytest


def pytest_configure(config):
    if (
        os.environ.get("TSECON_REQUIRE_DATAECON") == "1"
        and importlib.util.find_spec("tsecon.dataecon._native") is None
    ):
        raise pytest.UsageError(
            "DataEcon native support is required for this run but the extension was not built."
        )
