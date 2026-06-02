"""Text-based nutrition resolver with LLM-assisted food canonicalization."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from agents.vision.nutrition_lookup import NutritionLookup
from schemas import (
    AgentResponse,
    NutritionEstimate,
    TextNutritionAgentData,
    TextNutritionInputSummary,
    TextNutritionItemEstimate,
)


class TextNutritionAgent(BaseAgent):
    """Converts text-extracted foods and quantities into nutrition estimates."""

    DEFAULT_MASS_G = {
        "apple": 180.0,
        "almond": 30.0,
        "bread": 30.0,
        "cheese": 50.0,
        "chicken": 150.0,
        "egg": 50.0,
        "feta cheese": 50.0,
        "french fries": 120.0,
        "olive oil": 13.5,
        "pasta": 180.0,
        "potato": 180.0,
        "rice": 180.0,
        "sausage": 75.0,
        "spinach": 100.0,
        "tomato": 120.0,
        "yogurt": 150.0,
    }

    UNIT_TO_GRAMS = {
        "kg": 1000.0,
        "kilo": 1000.0,
        "kilogram": 1000.0,
        "g": 1.0,
        "gr": 1.0,
        "gram": 1.0,
        "gramlık": 1.0,
    }

    COUNT_UNITS = {
        "adet",
        "tane",
        "dilim",
        "porsiyon",
        "kaşık",
        "kasik",
        "yemek kaşığı",
        "tatlı kaşığı",
    }

    def __init__(
        self,
        canonicalization_model: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        self.nutrition_lookup = NutritionLookup()
        self.canonicalization_model = (
            canonicalization_model
            or os.getenv("OLLAMA_FOOD_MODEL")
            or os.getenv("OLLAMA_FUSION_MODEL")
            or "qwen3.5:9b"
        )
        self.timeout_seconds = timeout_seconds or int(
            os.getenv("OLLAMA_FOOD_TIMEOUT_SECONDS", "15")
        )
        self._canonicalization_cache: dict[str, dict[str, Any]] = {}
        self.food_aliases = self._load_shared_aliases()

    async def process(self, input_data: Any) -> AgentResponse:
        payload = input_data if isinstance(input_data, dict) else {}
        text_response = self._coerce_response(payload.get("text"))
        vision_response = self._coerce_response(payload.get("vision"))

        text_data = text_response.data if text_response else {}
        raw_text = (text_data.get("input") or {}).get("raw_text") or payload.get("raw_text") or ""
        vision_mass_g = ((vision_response.data if vision_response else {}).get("totals") or {}).get("mass_g")

        assumptions: list[str] = []
        warnings: list[str] = []
        foods, canonicalized_foods, canonicalization_warnings = await self._extract_and_canonicalize_foods(
            text_data=text_data,
            raw_text=raw_text,
        )
        warnings.extend(canonicalization_warnings)
        if not foods:
            contract = TextNutritionAgentData(
                input=TextNutritionInputSummary(raw_text=raw_text, vision_mass_g=vision_mass_g),
                warnings=["No usable food names were found in text."],
            )
            return AgentResponse(
                source="text_nutrition",
                confidence=0.0,
                data=contract.model_dump(),
                error="No usable food names were found in text.",
            )

        items = self._build_items(
            foods=foods,
            canonicalized_foods=canonicalized_foods,
            raw_text=raw_text,
            text_data=text_data,
            vision_mass_g=vision_mass_g,
            assumptions=assumptions,
            warnings=warnings,
        )
        totals = self._sum_items(items)
        contract = TextNutritionAgentData(
            input=TextNutritionInputSummary(raw_text=raw_text, vision_mass_g=vision_mass_g),
            items=items,
            totals=totals,
            assumptions=assumptions,
            warnings=warnings,
            llm_notes=[
                "Text nutrition is deterministic and uses nutrition priors; the LLM should not calculate macros.",
                "Food spans and canonical English nutrition names are produced by the local LLM before nutrition lookup when available.",
                "Prefer explicit gram quantities over default or vision-distributed mass assumptions.",
            ],
        )
        return AgentResponse(
            source="text_nutrition",
            confidence=self._confidence(items),
            data=contract.model_dump(),
            error=None,
        )

    def _coerce_response(self, value: Any) -> AgentResponse | None:
        if value is None:
            return None
        if isinstance(value, AgentResponse):
            return value
        if isinstance(value, dict):
            return AgentResponse.model_validate(value)
        return None

    async def _extract_and_canonicalize_foods(
        self,
        text_data: dict[str, Any],
        raw_text: str,
    ) -> tuple[list[str], dict[str, dict[str, Any]], list[str]]:
        parsed_foods = self._foods_from_text(text_data, raw_text)
        warnings: list[str] = []

        if raw_text.strip():
            try:
                llm_values = await self._extract_and_canonicalize_with_llm(
                    raw_text=raw_text,
                    parsed_foods=parsed_foods,
                )
                self._canonicalization_cache.update(llm_values)
                foods = self._ordered_foods_from_canonicalized(llm_values)
                if foods:
                    return foods, {
                        self._normalize(food): self._canonicalization_cache[self._normalize(food)]
                        for food in foods
                    }, warnings
            except Exception as exc:
                warnings.append(f"Food extraction/canonicalization LLM failed; fallback used: {exc}")

        canonicalized_foods, canonicalization_warnings = await self._canonicalize_foods(parsed_foods)
        warnings.extend(canonicalization_warnings)
        return parsed_foods, canonicalized_foods, warnings

    def _foods_from_text(self, text_data: dict[str, Any], raw_text: str) -> list[str]:
        parsed = text_data.get("parsed") or {}
        candidates = list(parsed.get("foods") or [])

        if not candidates and raw_text:
            candidates.extend(
                part.strip()
                for part in re.split(r"\s*(?:,|&|\+|\bve\b|\band\b)\s*", raw_text, flags=re.IGNORECASE)
            )

        foods: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            cleaned = self._strip_quantity_units(str(candidate).strip(" .;:"))
            normalized = self._normalize(cleaned)
            if len(normalized) < 3:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            foods.append(cleaned)
        return foods

    def _strip_quantity_units(self, value: str) -> str:
        units = "kg|kilo|kilogram|g|gr|gram|gramlık|adet|tane|dilim|porsiyon|kaşık|kasik"
        cleaned = re.sub(
            rf"^\s*\d+(?:[.,]\d+)?\s*(?:{units})\s+",
            "",
            value,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            rf"\s+\d+(?:[.,]\d+)?\s*(?:{units})\s*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned.strip(" .;:")

    def _ordered_foods_from_canonicalized(
        self,
        canonicalized: dict[str, dict[str, Any]],
    ) -> list[str]:
        ordered = sorted(
            canonicalized.items(),
            key=lambda item: int(item[1].get("order", 0)),
        )
        foods: list[str] = []
        seen: set[str] = set()
        for key, value in ordered:
            display_name = str(value.get("display_name") or key).strip()
            normalized = self._normalize(display_name)
            if len(normalized) < 3 or normalized in seen:
                continue
            seen.add(normalized)
            foods.append(display_name)
        return foods

    def _build_items(
        self,
        foods: list[str],
        canonicalized_foods: dict[str, dict[str, Any]],
        raw_text: str,
        text_data: dict[str, Any],
        vision_mass_g: float | None,
        assumptions: list[str],
        warnings: list[str],
    ) -> list[TextNutritionItemEstimate]:
        explicit_masses = {
            self._normalize(food): self._explicit_mass_for_food(
                raw_text,
                food,
                canonicalized_foods.get(self._normalize(food), {}).get("canonical_name"),
            )
            for food in foods
        }
        explicit_count = sum(1 for value in explicit_masses.values() if value is not None)
        missing_count = len(foods) - explicit_count
        remaining_vision_mass = None
        if vision_mass_g is not None and missing_count > 0:
            explicit_total = sum(value or 0.0 for value in explicit_masses.values())
            remaining_vision_mass = max(float(vision_mass_g) - explicit_total, 0.0)

        items: list[TextNutritionItemEstimate] = []
        entity_confidence = self._entity_confidence(text_data)
        for food in foods:
            canonicalized = canonicalized_foods.get(self._normalize(food), {})
            lookup_name = str(canonicalized.get("canonical_name") or self._lookup_name(food))
            canonicalization_source = str(canonicalized.get("source") or "raw")
            mass_g = explicit_masses[self._normalize(food)]
            mass_source = "explicit_text"
            item_assumptions: list[str] = []

            if mass_g is None and remaining_vision_mass is not None:
                mass_g = remaining_vision_mass / missing_count
                mass_source = "vision_distributed"
                item_assumptions.append("Mass was distributed from vision total mass because text did not provide grams.")
            elif mass_g is None:
                mass_g = self.DEFAULT_MASS_G.get(lookup_name, 100.0)
                mass_source = "default_serving"
                item_assumptions.append("Default serving mass was used because text did not provide grams and vision mass was unavailable.")

            if item_assumptions:
                assumptions.extend(f"{food}: {assumption}" for assumption in item_assumptions)

            prior = self.nutrition_lookup.lookup(lookup_name)
            if not prior.matched:
                warnings.append(f"{food}: no close nutrition prior match; generic prior used.")

            nutrition = NutritionEstimate(
                calories_kcal=mass_g * prior.calories_per_g,
                mass_g=mass_g,
                protein_g=mass_g * prior.protein_per_g,
                carbs_g=mass_g * prior.carbs_per_g,
                fat_g=mass_g * prior.fat_per_g,
            )
            confidence = entity_confidence.get(
                self._normalize(food),
                float(canonicalized.get("confidence") or 0.7),
            )
            if mass_source != "explicit_text":
                confidence -= 0.15
            if not prior.matched:
                confidence -= 0.2
            items.append(
                TextNutritionItemEstimate(
                    name=food,
                    lookup_name=lookup_name,
                    canonicalization_source=canonicalization_source,
                    mass_g=mass_g,
                    mass_source=mass_source,
                    nutrition=nutrition,
                    confidence=max(min(confidence, 0.95), 0.1),
                    assumptions=item_assumptions,
                )
            )
        return items

    def _explicit_mass_for_food(
        self,
        raw_text: str,
        food: str,
        lookup_name: str | None = None,
    ) -> float | None:
        lookup_name = lookup_name or self._lookup_name(food)
        names = {food, lookup_name}
        names.update(
            turkish
            for turkish, english in self.food_aliases.items()
            if english == lookup_name
        )
        lowered = raw_text.lower()
        for name in sorted(names, key=len, reverse=True):
            escaped = re.escape(name.lower())
            before = re.search(
                rf"(\d+(?:[.,]\d+)?)\s*(kg|kilo|kilogram|g|gr|gram|gramlık|adet|tane|dilim|porsiyon|kaşık|kasik)\s+{escaped}\b",
                lowered,
            )
            after = re.search(
                rf"\b{escaped}\s+(\d+(?:[.,]\d+)?)\s*(kg|kilo|kilogram|g|gr|gram|gramlık|adet|tane|dilim|porsiyon|kaşık|kasik)\b",
                lowered,
            )
            match = before or after
            if not match:
                continue
            quantity = float(match.group(1).replace(",", "."))
            unit = match.group(2)
            if unit in self.UNIT_TO_GRAMS:
                return quantity * self.UNIT_TO_GRAMS[unit]
            if unit in self.COUNT_UNITS:
                return quantity * self.DEFAULT_MASS_G.get(lookup_name, 100.0)
        return None

    def _lookup_name(self, food: str) -> str:
        normalized = self._normalize(food)
        return self.food_aliases.get(normalized, normalized)

    async def _canonicalize_foods(
        self,
        foods: list[str],
    ) -> tuple[dict[str, dict[str, Any]], list[str]]:
        uncached = [
            food
            for food in foods
            if self._normalize(food) not in self._canonicalization_cache
        ]
        warnings: list[str] = []

        if uncached:
            try:
                llm_values = await self._canonicalize_with_llm(uncached)
                self._canonicalization_cache.update(llm_values)
            except Exception as exc:
                warnings.append(f"Food canonicalization LLM failed; alias/raw fallback used: {exc}")

            for food in uncached:
                key = self._normalize(food)
                if key not in self._canonicalization_cache:
                    fallback = self.food_aliases.get(key)
                    self._canonicalization_cache[key] = {
                        "canonical_name": fallback or key,
                        "confidence": 0.55 if fallback else 0.35,
                        "source": "alias_fallback" if fallback else "raw",
                    }

        return {
            self._normalize(food): self._canonicalization_cache[self._normalize(food)]
            for food in foods
        }, warnings

    async def _canonicalize_with_llm(self, foods: list[str]) -> dict[str, dict[str, Any]]:
        return await self._extract_and_canonicalize_with_llm(raw_text="", parsed_foods=foods)

    async def _extract_and_canonicalize_with_llm(
        self,
        raw_text: str,
        parsed_foods: list[str],
    ) -> dict[str, dict[str, Any]]:
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError(
                "langchain, langchain-core, and langchain-ollama are required for food canonicalization."
            ) from exc

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You extract food ingredient spans from Turkish or English meal text and normalize them to "
                    "canonical English food names for a nutrition database. Return only JSON. Do not calculate "
                    "calories, macros, grams, quantities, or portions. Do not include quantities or units in "
                    "input_name; input_name must be the food phrase as written by the user. Use common lowercase "
                    "English food names for canonical_name. If a parsed food list is provided, include those foods "
                    "unless they are clearly not food. If unsure, use the closest common food name and lower confidence.",
                ),
                (
                    "human",
                    "Raw text:\n{raw_text}\n\n"
                    "Parsed food candidates JSON array:\n{foods_json}\n\n"
                    "Return JSON exactly in this shape: "
                    "{{\"items\":[{{\"input_name\":\"...\",\"canonical_name\":\"...\",\"confidence\":0.0,\"notes\":[]}}]}}"
                ),
            ]
        )
        llm = ChatOllama(
            model=self.canonicalization_model,
            temperature=0.0,
            reasoning=False,
            format="json",
            num_predict=256,
            async_client_kwargs={"timeout": self.timeout_seconds},
        )
        result = await (prompt | llm).ainvoke(
            {
                "raw_text": raw_text,
                "foods_json": json.dumps(parsed_foods, ensure_ascii=False),
            }
        )
        content = result.content if hasattr(result, "content") else result
        if isinstance(content, list):
            content = "".join(str(part) for part in content)
        data = json.loads(str(content))
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise ValueError("LLM canonicalization returned JSON without an items list.")

        canonicalized: dict[str, dict[str, Any]] = {}
        requested = {self._normalize(food) for food in parsed_foods}
        for order, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            input_name = self._normalize(str(item.get("input_name") or ""))
            canonical_name = self._normalize(str(item.get("canonical_name") or ""))
            if not input_name or not canonical_name:
                continue
            if not raw_text and requested and input_name not in requested:
                requested_match = self._closest_requested_name(input_name, requested)
                if requested_match is None:
                    continue
                input_name = requested_match
            if len(input_name) < 3:
                continue
            try:
                confidence = max(min(float(item.get("confidence", 0.75)), 1.0), 0.0)
            except (TypeError, ValueError):
                confidence = 0.75
            canonicalized[input_name] = {
                "canonical_name": canonical_name,
                "confidence": confidence,
                "display_name": str(item.get("input_name") or input_name).strip(),
                "source": "llm",
                "order": order,
            }

        if not canonicalized:
            raise ValueError("LLM canonicalization did not return usable food names.")
        return canonicalized

    def _closest_requested_name(self, input_name: str, requested: set[str]) -> str | None:
        for candidate in requested:
            if input_name == candidate or input_name in candidate or candidate in input_name:
                return candidate
        return None

    def _entity_confidence(self, text_data: dict[str, Any]) -> dict[str, float]:
        confidences: dict[str, float] = {}
        for entity in text_data.get("entities") or []:
            if entity.get("type") != "food":
                continue
            normalized = self._normalize(str(entity.get("text", "")))
            if not normalized:
                continue
            confidences[normalized] = max(
                confidences.get(normalized, 0.0),
                float(entity.get("confidence") or 0.0),
            )
        return confidences

    def _sum_items(self, items: list[TextNutritionItemEstimate]) -> NutritionEstimate:
        return NutritionEstimate(
            calories_kcal=sum(item.nutrition.calories_kcal or 0.0 for item in items),
            mass_g=sum(item.nutrition.mass_g or 0.0 for item in items),
            protein_g=sum(item.nutrition.protein_g or 0.0 for item in items),
            carbs_g=sum(item.nutrition.carbs_g or 0.0 for item in items),
            fat_g=sum(item.nutrition.fat_g or 0.0 for item in items),
        )

    def _confidence(self, items: list[TextNutritionItemEstimate]) -> float:
        if not items:
            return 0.0
        return sum(item.confidence for item in items) / len(items)

    def _normalize(self, value: str) -> str:
        return " ".join(value.strip().lower().replace("_", " ").split())

    def _load_shared_aliases(self) -> dict[str, str]:
        path = Path(__file__).resolve().parents[1] / "food" / "aliases.json"
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            self._normalize(str(key)): self._normalize(str(value))
            for key, value in data.items()
        }
