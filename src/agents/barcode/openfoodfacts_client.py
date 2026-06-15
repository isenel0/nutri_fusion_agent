"""HTTP client and response normalization for OpenFoodFacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from config import Settings


class OpenFoodFactsClient:
    """Fetch and normalize product data from OpenFoodFacts."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def get_product(self, barcode: str) -> dict[str, Any]:
        cached = self.get_cached_product(barcode)
        if cached is not None:
            return cached

        url = f"{self.settings.openfoodfacts_base_url}/product/{barcode}.json"
        try:
            response = requests.get(
                url,
                timeout=self.settings.openfoodfacts_timeout_seconds,
                headers={"User-Agent": self.settings.openfoodfacts_user_agent},
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise ConnectionError(f"OpenFoodFacts request failed: {exc}") from exc
        except ValueError as exc:
            raise ValueError("OpenFoodFacts returned invalid JSON payload") from exc

        if payload.get("status") != 1:
            raise LookupError(
                payload.get("status_verbose", "Product not found in OpenFoodFacts.")
            )

        return payload

    def get_cached_product(self, barcode: str) -> dict[str, Any] | None:
        cache_path = self._cache_path()
        if cache_path is None or not cache_path.exists():
            return None
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        products = cache.get("products") if isinstance(cache, dict) else None
        record = products.get(barcode) if isinstance(products, dict) else None
        if not isinstance(record, dict):
            return None
        return self._cached_record_to_payload(barcode, record)

    def _cache_path(self) -> Path | None:
        configured = getattr(self.settings, "barcode_nutrition_cache_path", None)
        if not configured:
            return None
        path = Path(configured).expanduser()
        if path.is_absolute() or path.exists():
            return path

        repo_root = Path(__file__).resolve().parents[3]
        candidates = [
            repo_root / path,
            repo_root.parent / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return path

    def _cached_record_to_payload(self, barcode: str, record: dict[str, Any]) -> dict[str, Any]:
        nutrition = record.get("nutrition_per_100g") or {}
        nutriments = {
            "energy-kcal_100g": nutrition.get("calories_kcal"),
            "proteins_100g": nutrition.get("protein_g"),
            "carbohydrates_100g": nutrition.get("carbs_g"),
            "fat_100g": nutrition.get("fat_g"),
            "fiber_100g": nutrition.get("fiber_g"),
            "sugars_100g": nutrition.get("sugar_g"),
            "sodium_100g": nutrition.get("sodium_g"),
        }
        return {
            "status": 1,
            "product": {
                "code_type": record.get("code_type") or "local_cache",
                "product_name": record.get("product_name"),
                "brands": record.get("brand"),
                "categories": record.get("categories"),
                "quantity": record.get("quantity"),
                "serving_size": record.get("serving_size"),
                "nutriments": nutriments,
                "_local_cache_record": record,
            },
        }

    def to_standard_payload(
        self,
        barcode: str,
        payload: dict[str, Any],
        include_raw_product: bool = False,
    ) -> dict[str, Any]:
        product = payload.get("product", {})
        nutriments = product.get("nutriments", {})

        structured: dict[str, Any] = {
            "barcode": {
                "value": barcode,
                "type": product.get("code_type") or "unknown",
            },
            "product": {
                "name": product.get("product_name"),
                "brand": product.get("brands"),
                "categories": product.get("categories"),
                "quantity": product.get("quantity"),
                "serving_size": product.get("serving_size"),
            },
            "nutrition_per_100g": {
                "calories_kcal": _as_float(
                    nutriments.get("energy-kcal_100g")
                    or nutriments.get("energy-kcal_value")
                ),
                "protein_g": _as_float(nutriments.get("proteins_100g")),
                "carbs_g": _as_float(nutriments.get("carbohydrates_100g")),
                "fat_g": _as_float(nutriments.get("fat_100g")),
                "fiber_g": _as_float(nutriments.get("fiber_100g")),
                "sugar_g": _as_float(nutriments.get("sugars_100g")),
                "sodium_g": _as_float(nutriments.get("sodium_100g")),
            },
            "source": {
                "provider": (
                    "local_barcode_cache"
                    if product.get("_local_cache_record")
                    else "openfoodfacts"
                ),
                "fetched_barcode": barcode,
            },
        }

        if include_raw_product:
            structured["raw_product"] = product

        return structured


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
