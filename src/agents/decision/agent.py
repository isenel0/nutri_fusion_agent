"""Local LLM decision agent for fusion planning."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from agents.base_agent import BaseAgent
from agents.decision.prompts import SYSTEM_PROMPT
from schemas import AgentResponse, FusionDecision


class FusionDecisionAgent(BaseAgent):
    """Uses local Qwen/Ollama via LangChain to choose fusion rules.

    The returned decision is advisory. FusionAgent still validates availability
    of barcode/vision numeric sources before computing final nutrition.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        timeout_seconds: int | None = None,
    ) -> None:
        self.model = model or os.getenv("OLLAMA_FUSION_MODEL", "qwen3.5:9b")
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds or int(
            os.getenv("OLLAMA_FUSION_TIMEOUT_SECONDS", "30")
        )

    async def process(self, input_data: Any) -> AgentResponse:
        context = self._extract_context(input_data)
        fallback = self._fallback_decision(context)

        try:
            decision = await self._call_llm(context)
            confidence = 0.75
            error = None
        except Exception as exc:
            decision = fallback
            confidence = 0.25
            error = f"LLM fusion decision failed; used deterministic fallback: {exc}"

        return AgentResponse(
            source="fusion_decision",
            confidence=confidence,
            data=decision.model_dump(),
            error=error,
        )

    async def _call_llm(self, context: dict[str, Any]) -> FusionDecision:
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError(
                "langchain, langchain-core, and langchain-ollama are required."
            ) from exc

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "Evidence JSON:\n{context_json}\n\n"
                    "Return JSON with exactly these keys: "
                    "primary_nutrition_source, mass_source, portion_multiplier, "
                    "corrected_food_names, hidden_ingredients, applied_rules, "
                    "conflicts, assumptions, explanation.",
                ),
            ]
        )
        llm = ChatOllama(
            model=self.model,
            temperature=self.temperature,
            reasoning=False,
            format="json",
            num_predict=512,
            async_client_kwargs={"timeout": self.timeout_seconds},
        )
        chain = prompt | llm
        result = await chain.ainvoke(
            {
                "context_json": json.dumps(
                    context,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            }
        )
        content = result.content if hasattr(result, "content") else result
        if isinstance(content, list):
            content = "".join(str(part) for part in content)
        raw_decision = json.loads(str(content))
        return FusionDecision.model_validate(self._normalize_decision(raw_decision))

    def _extract_context(self, input_data: Any) -> dict[str, Any]:
        data = input_data.data if isinstance(input_data, AgentResponse) else input_data
        if not isinstance(data, dict):
            return {}
        if data.get("contract_version") == "orchestrator.v1":
            return data.get("llm_context") or {}
        return data.get("llm_context") or data

    def _fallback_decision(self, context: dict[str, Any]) -> FusionDecision:
        barcode = context.get("barcode", {}).get("summary", {})
        vision = context.get("vision", {}).get("summary", {})
        text = context.get("text", {}).get("summary", {})
        text_nutrition = context.get("text_nutrition", {}).get("summary", {})
        ingredient_resolution = context.get("ingredient_resolution", {}).get("summary", {})

        barcode_nutrition = barcode.get("nutrition_per_100g") or {}
        vision_totals = vision.get("totals") or {}
        text_nutrition_totals = text_nutrition.get("totals") or {}
        ingredient_resolution_totals = ingredient_resolution.get("totals") or {}
        parsed = text.get("parsed") or {}

        if self._has_macros(barcode_nutrition):
            source = "barcode"
            mass_source = "vision" if vision_totals.get("mass_g") is not None else "assumed_100g"
        elif self._has_macros(ingredient_resolution_totals):
            source = "ingredient_resolution"
            mass_source = "vision" if vision_totals.get("mass_g") is not None else "text"
        elif self._has_macros(text_nutrition_totals) and self._has_explicit_text_mass(text_nutrition):
            source = "text_nutrition"
            mass_source = "text"
        elif self._has_macros(vision_totals):
            source = "vision"
            mass_source = "vision"
        elif self._has_macros(text_nutrition_totals):
            source = "text_nutrition"
            mass_source = "text"
        else:
            source = "fallback"
            mass_source = "unknown"

        return FusionDecision(
            primary_nutrition_source=source,
            mass_source=mass_source,
            portion_multiplier=float(parsed.get("portion_multiplier") or 1.0),
            corrected_food_names=self._fallback_name_overrides(context),
            applied_rules=["fallback_fusion_decision"],
            assumptions=[],
            conflicts=[],
            explanation="Deterministic fallback decision generated without LLM reasoning.",
        )

    def _normalize_decision(self, value: Any) -> dict[str, Any]:
        decision = value if isinstance(value, dict) else {}

        if decision.get("primary_nutrition_source") not in {"barcode", "vision", "text_nutrition", "ingredient_resolution", "fallback"}:
            decision["primary_nutrition_source"] = "fallback"

        if decision.get("mass_source") not in {"vision", "text", "assumed_100g", "unknown"}:
            if decision.get("primary_nutrition_source") == "barcode":
                decision["mass_source"] = "assumed_100g"
            elif decision.get("primary_nutrition_source") == "vision":
                decision["mass_source"] = "vision"
            elif decision.get("primary_nutrition_source") == "text_nutrition":
                decision["mass_source"] = "text"
            elif decision.get("primary_nutrition_source") == "ingredient_resolution":
                decision["mass_source"] = "vision"
            else:
                decision["mass_source"] = "unknown"

        try:
            decision["portion_multiplier"] = max(float(decision.get("portion_multiplier", 1.0)), 0.0)
        except (TypeError, ValueError):
            decision["portion_multiplier"] = 1.0

        corrected = decision.get("corrected_food_names")
        if not isinstance(corrected, dict):
            decision["corrected_food_names"] = {}

        for key in ("hidden_ingredients", "applied_rules", "conflicts", "assumptions"):
            if not isinstance(decision.get(key), list):
                decision[key] = []

        if decision.get("explanation") is not None:
            decision["explanation"] = str(decision["explanation"])

        return decision

    def _has_macros(self, nutrition: dict[str, Any]) -> bool:
        return any(
            nutrition.get(key) is not None
            for key in ("calories_kcal", "protein_g", "carbs_g", "fat_g")
        )

    def _has_explicit_text_mass(self, text_nutrition_summary: dict[str, Any]) -> bool:
        items = text_nutrition_summary.get("items") or []
        return bool(items) and all(item.get("mass_source") == "explicit_text" for item in items)

    def _fallback_name_overrides(self, context: dict[str, Any]) -> dict[str, str]:
        vision = context.get("vision", {}).get("summary", {})
        text = context.get("text", {}).get("summary", {})
        barcode = context.get("barcode", {}).get("summary", {})

        text_foods = self._text_foods(text)
        if not text_foods:
            return {}

        vision_names = [
            str(ingredient.get("name"))
            for ingredient in vision.get("ingredients", [])
            if ingredient.get("name")
        ]
        if vision_names:
            vision_tokens = {self._normalize_food_name(name) for name in vision_names}
            text_tokens = {self._normalize_food_name(name) for name in text_foods}
            if vision_tokens & text_tokens:
                return {}
            if len(vision_names) == len(text_foods):
                return dict(zip(vision_names, text_foods))

        product = barcode.get("product") or {}
        product_name = product.get("name")
        if product_name and len(text_foods) == 1:
            return {str(product_name): text_foods[0]}

        return {}

    def _text_foods(self, text_summary: dict[str, Any]) -> list[str]:
        parsed = text_summary.get("parsed") or {}
        candidates = parsed.get("foods") or []
        if not candidates:
            raw_text = text_summary.get("raw_text") or ""
            candidates = [raw_text]

        foods: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            for part in re.split(r"\s*(?:,|&|\band\b|\bve\b|\+)\s*", str(candidate), flags=re.IGNORECASE):
                cleaned = part.strip(" .;:")
                normalized = self._normalize_food_name(cleaned)
                if cleaned and normalized and normalized not in seen:
                    foods.append(cleaned)
                    seen.add(normalized)
        return foods

    def _normalize_food_name(self, name: str) -> str:
        return re.sub(r"[^a-z0-9çğıöşü]+", " ", name.lower()).strip()
