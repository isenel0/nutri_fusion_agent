"""Barcode agent implementation for barcode decoding and product lookup."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from agents.base_agent import BaseAgent
from agents.barcode.openfoodfacts_client import OpenFoodFactsClient
from agents.barcode.service import BarcodeScanner
from config import Settings
from schemas import AgentResponse


class BarcodeAgent(BaseAgent):
    """Extract barcode and fetch packaged food nutrition details."""

    BARCODE_PATTERN = re.compile(r"^[A-Za-z0-9]{3,30}$")

    def __init__(
        self,
        settings: Settings | None = None,
        scanner: BarcodeScanner | None = None,
        nutrition_client: OpenFoodFactsClient | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.scanner = scanner or BarcodeScanner()
        self.nutrition_client = nutrition_client or OpenFoodFactsClient(self.settings)

    async def process(self, input_data: Any) -> AgentResponse:
        """Resolve barcode from input and return standardized nutrition payload.

        Supported input_data formats:
        - barcode string: "5449000000996"
        - dict payload with optional keys:
          barcode, image_path, image_bytes/bytes, image_filename
        """
        try:
            resolved = await self._resolve_input(input_data)
        except Exception as exc:
            return AgentResponse(
                source="barcode",
                confidence=self.settings.barcode_confidence_on_error,
                data={
                    "input": {"raw": str(input_data)},
                    "barcode": None,
                    "product": None,
                    "nutrition_per_100g": None,
                },
                error=f"Barcode input resolution failed: {exc}",
            )

        barcode_value = resolved.get("barcode")
        input_meta = resolved.get("input", {})

        if not barcode_value:
            return AgentResponse(
                source="barcode",
                confidence=self.settings.barcode_confidence_on_error,
                data={
                    "input": input_meta,
                    "barcode": None,
                    "product": None,
                    "nutrition_per_100g": None,
                },
                error="No barcode could be resolved from the provided input.",
            )

        try:
            payload = await asyncio.to_thread(self.nutrition_client.get_product, barcode_value)
            standardized = self.nutrition_client.to_standard_payload(
                barcode=barcode_value,
                payload=payload,
                include_raw_product=self.settings.include_raw_product_payload,
            )
        except Exception as exc:
            return AgentResponse(
                source="barcode",
                confidence=self.settings.barcode_confidence_without_macros,
                data={
                    "input": input_meta,
                    "barcode": {"value": barcode_value},
                    "product": None,
                    "nutrition_per_100g": None,
                },
                error=f"Barcode detected, but nutrition lookup failed: {exc}",
            )

        has_any_macro = any(
            standardized.get("nutrition_per_100g", {}).get(key) is not None
            for key in ("calories_kcal", "protein_g", "carbs_g", "fat_g")
        )

        confidence = (
            self.settings.barcode_confidence_with_macros
            if has_any_macro
            else self.settings.barcode_confidence_without_macros
        )

        standardized["input"] = input_meta
        standardized["reasoning_summary"] = (
            "Resolved barcode from explicit value or image decoding, then mapped "
            "OpenFoodFacts fields into a standard nutrition_per_100g payload."
        )

        return AgentResponse(
            source="barcode",
            confidence=confidence,
            data=standardized,
            error=None,
        )

    async def _resolve_input(self, input_data: Any) -> dict[str, Any]:
        if isinstance(input_data, str):
            barcode = self._normalize_barcode(input_data)
            return {
                "barcode": barcode,
                "input": {
                    "barcode_text": input_data,
                    "image_filename": None,
                },
            }

        if not isinstance(input_data, dict):
            raise TypeError("barcode input must be a string or dictionary")

        explicit_barcode = self._normalize_barcode(input_data.get("barcode"))
        image_path = input_data.get("image_path")
        image_bytes = input_data.get("image_bytes") or input_data.get("bytes")
        image_filename = input_data.get("image_filename") or input_data.get("filename")

        if explicit_barcode:
            return {
                "barcode": explicit_barcode,
                "input": {
                    "barcode_text": input_data.get("barcode"),
                    "image_filename": image_filename,
                },
            }

        if image_path or image_bytes is not None:
            scanned_barcode = await asyncio.to_thread(
                self.scanner.scan,
                image_path,
                image_bytes,
            )
            return {
                "barcode": self._normalize_barcode(scanned_barcode),
                "input": {
                    "barcode_text": None,
                    "image_filename": image_filename,
                },
            }

        return {
            "barcode": None,
            "input": {
                "barcode_text": input_data.get("barcode"),
                "image_filename": image_filename,
            },
        }

    def _normalize_barcode(self, value: Any) -> str | None:
        if value is None:
            return None
        barcode = str(value).strip()
        if not barcode:
            return None
        if not self.BARCODE_PATTERN.match(barcode):
            raise ValueError(f"Invalid barcode format: {barcode}")
        return barcode
