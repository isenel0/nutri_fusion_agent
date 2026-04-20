"""HTTP client and response normalization for OpenFoodFacts."""

from __future__ import annotations

from typing import Any

import requests

from config import Settings


class OpenFoodFactsClient:
    """Fetch and normalize product data from OpenFoodFacts."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def get_product(self, barcode: str) -> dict[str, Any]:
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
                "provider": "openfoodfacts",
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
