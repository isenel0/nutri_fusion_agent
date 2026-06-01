"""Barcode agent implementation for barcode decoding and product lookup."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from agents.base_agent import BaseAgent
from agents.barcode.openfoodfacts_client import OpenFoodFactsClient
from agents.barcode.service import BarcodeScanner
from config import Settings
from schemas import (
    AgentResponse,
    BarcodeAgentData,
    BarcodeIdentity,
    BarcodeInputSummary,
    BarcodeProduct,
    BarcodeSource,
    NutritionEstimate,
)


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
            contract = self._empty_contract(
                input_meta={"barcode_text": None, "image_filename": None},
                notes=["Barcode input could not be normalized or decoded."],
            )
            return AgentResponse(
                source="barcode",
                confidence=self.settings.barcode_confidence_on_error,
                data=contract.model_dump(),
                error=f"Barcode input resolution failed: {exc}",
            )

        barcode_value = resolved.get("barcode")
        input_meta = resolved.get("input", {})

        if not barcode_value:
            contract = self._empty_contract(
                input_meta=input_meta,
                notes=["No barcode was available from text or image input."],
            )
            return AgentResponse(
                source="barcode",
                confidence=self.settings.barcode_confidence_on_error,
                data=contract.model_dump(),
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
            contract = self._empty_contract(
                input_meta=input_meta,
                barcode_value=barcode_value,
                notes=[
                    "A barcode was resolved, but product nutrition lookup failed.",
                    "Do not infer packaged-food macros from this barcode response.",
                ],
            )
            return AgentResponse(
                source="barcode",
                confidence=self.settings.barcode_confidence_without_macros,
                data=contract.model_dump(),
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

        contract = self._contract_from_lookup(
            standardized=standardized,
            input_meta=input_meta,
            notes=[
                "Resolved barcode from explicit value or image decoding.",
                "Mapped OpenFoodFacts fields into nutrition_per_100g for deterministic macro math.",
            ],
        )

        return AgentResponse(
            source="barcode",
            confidence=confidence,
            data=contract.model_dump(),
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

    def _empty_contract(
        self,
        input_meta: dict[str, Any],
        notes: list[str],
        barcode_value: str | None = None,
    ) -> BarcodeAgentData:
        return BarcodeAgentData(
            input=self._input_summary(input_meta),
            barcode=(
                BarcodeIdentity(value=barcode_value, type=None)
                if barcode_value
                else None
            ),
            product=None,
            nutrition_per_100g=None,
            source=BarcodeSource(fetched_barcode=barcode_value),
            llm_notes=notes,
        )

    def _contract_from_lookup(
        self,
        standardized: dict[str, Any],
        input_meta: dict[str, Any],
        notes: list[str],
    ) -> BarcodeAgentData:
        barcode = standardized.get("barcode") or {}
        product = standardized.get("product") or {}
        nutrition = standardized.get("nutrition_per_100g") or {}
        source = standardized.get("source") or {}
        barcode_value = barcode.get("value") or source.get("fetched_barcode")

        return BarcodeAgentData(
            input=self._input_summary(input_meta),
            barcode=BarcodeIdentity(
                value=str(barcode_value),
                type=barcode.get("type"),
            ),
            product=BarcodeProduct(
                name=product.get("name"),
                brand=product.get("brand"),
                categories=product.get("categories"),
                quantity=product.get("quantity"),
                serving_size=product.get("serving_size"),
            ),
            nutrition_per_100g=NutritionEstimate(
                calories_kcal=nutrition.get("calories_kcal"),
                protein_g=nutrition.get("protein_g"),
                carbs_g=nutrition.get("carbs_g"),
                fat_g=nutrition.get("fat_g"),
                fiber_g=nutrition.get("fiber_g"),
                sugar_g=nutrition.get("sugar_g"),
                sodium_g=nutrition.get("sodium_g"),
            ),
            source=BarcodeSource(
                provider=source.get("provider", "openfoodfacts"),
                fetched_barcode=source.get("fetched_barcode"),
            ),
            llm_notes=notes,
            raw_product=standardized.get("raw_product"),
        )

    def _input_summary(self, input_meta: dict[str, Any]) -> BarcodeInputSummary:
        return BarcodeInputSummary(
            barcode_text=input_meta.get("barcode_text"),
            image_filename=input_meta.get("image_filename"),
        )
