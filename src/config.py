"""Centralized runtime configuration for the Nutri Fusion backend."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Environment-driven settings shared by all agents and API modules."""

    app_name: str = "nutri-fusion-agent"
    app_env: str = "dev"
    app_debug: bool = False

    openfoodfacts_base_url: str = "https://world.openfoodfacts.org/api/v2"
    openfoodfacts_timeout_seconds: int = 15
    openfoodfacts_user_agent: str = "nutri-fusion-agent/0.1"

    include_raw_product_payload: bool = False
    barcode_confidence_with_macros: float = 0.95
    barcode_confidence_without_macros: float = 0.75
    barcode_confidence_on_error: float = 0.0

    @staticmethod
    def from_env() -> "Settings":
        timeout = _safe_int(os.getenv("OPENFOODFACTS_TIMEOUT"), 15)

        return Settings(
            app_name=os.getenv("APP_NAME", "nutri-fusion-agent"),
            app_env=os.getenv("APP_ENV", "dev"),
            app_debug=_as_bool(os.getenv("APP_DEBUG"), default=False),
            openfoodfacts_base_url=os.getenv(
                "OPENFOODFACTS_BASE_URL",
                "https://world.openfoodfacts.org/api/v2",
            ),
            openfoodfacts_timeout_seconds=max(timeout, 1),
            openfoodfacts_user_agent=os.getenv(
                "OPENFOODFACTS_USER_AGENT",
                "nutri-fusion-agent/0.1",
            ),
            include_raw_product_payload=_as_bool(
                os.getenv("INCLUDE_RAW_PRODUCT_PAYLOAD"),
                default=False,
            ),
            barcode_confidence_with_macros=_safe_float(
                os.getenv("BARCODE_CONFIDENCE_WITH_MACROS"),
                0.95,
            ),
            barcode_confidence_without_macros=_safe_float(
                os.getenv("BARCODE_CONFIDENCE_WITHOUT_MACROS"),
                0.75,
            ),
            barcode_confidence_on_error=_safe_float(
                os.getenv("BARCODE_CONFIDENCE_ON_ERROR"),
                0.0,
            ),
        )


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _safe_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
