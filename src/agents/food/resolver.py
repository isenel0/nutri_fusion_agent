"""Ingredient identity resolver for cross-agent fusion."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from schemas import (
    AgentResponse,
    IngredientResolutionAgentData,
    IngredientSourceLink,
    NutritionEstimate,
    ResolvedIngredient,
)


class IngredientResolverAgent(BaseAgent):
    """Resolves raw vision/text ingredients into one calculation ingredient set."""

    FOOD_GROUPS = {
        "spinach": "leafy_green",
        "cilantro": "leafy_green",
        "mint": "leafy_green",
        "cilantro mint": "leafy_green",
        "lettuce": "leafy_green",
        "salad": "leafy_green",
        "parsley": "leafy_green",
        "rocket": "leafy_green",
        "arugula": "leafy_green",
        "rice": "grain",
        "pasta": "grain",
        "bread": "grain",
        "potato": "starch",
        "french fries": "starch",
        "cheese": "dairy",
        "feta cheese": "dairy",
        "yogurt": "dairy",
        "chicken": "meat",
        "beef": "meat",
        "pork": "meat",
        "fried meat": "meat",
        "sausage": "meat",
        "egg": "egg",
        "apple": "fruit",
        "banana": "fruit",
        "almond": "nut",
        "olives": "vegetable",
        "brussels sprouts": "vegetable",
        "cucumber": "vegetable",
        "tomato": "vegetable",
    }

    def __init__(
        self,
        aliases_path: Path | None = None,
        model: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        self.aliases_path = aliases_path or Path(__file__).with_name("aliases.json")
        self.aliases = self._load_aliases(self.aliases_path)
        self.model = (
            model
            or os.getenv("OLLAMA_INGREDIENT_MODEL")
            or os.getenv("OLLAMA_FOOD_MODEL")
            or os.getenv("OLLAMA_FUSION_MODEL")
            or "qwen3.5:9b"
        )
        self.timeout_seconds = timeout_seconds or int(
            os.getenv("OLLAMA_INGREDIENT_TIMEOUT_SECONDS", "20")
        )

    async def process(self, input_data: Any) -> AgentResponse:
        outputs = self._extract_outputs(input_data)
        vision = outputs.get("vision")
        text_nutrition = outputs.get("text_nutrition")
        decision = self._coerce_response(
            input_data.get("decision") if isinstance(input_data, dict) else None
        )

        vision_data = vision.data if vision else {}
        text_nutrition_data = text_nutrition.data if text_nutrition else {}
        decision_data = decision.data if decision else {}

        semantic_decisions: list[dict[str, Any]] = []
        semantic_warnings: list[str] = []
        if (vision_data.get("ingredients") or []) and (text_nutrition_data.get("items") or []):
            try:
                semantic_decisions = await self._llm_semantic_decisions(
                    vision_data=vision_data,
                    text_nutrition_data=text_nutrition_data,
                )
            except Exception as exc:
                semantic_warnings.append(
                    f"LLM ingredient resolution failed; deterministic resolver fallback used: {exc}"
                )

        items, replacements, unmatched_vision, unmatched_text, assumptions, warnings = (
            self._resolve(
                vision_data=vision_data,
                text_nutrition_data=text_nutrition_data,
                decision_data=decision_data,
                semantic_decisions=semantic_decisions,
            )
        )
        warnings.extend(semantic_warnings)
        contract = IngredientResolutionAgentData(
            items=items,
            totals=self._sum_items(items),
            replacements=replacements,
            unmatched_vision=unmatched_vision,
            unmatched_text=unmatched_text,
            assumptions=assumptions,
            warnings=warnings,
            llm_notes=[
                "This resolver decides ingredient identity and source linking; fusion still owns final result packaging.",
                "The local LLM may propose semantic match/replace/add/preserve decisions; deterministic validation applies the nutrition math.",
                "Explicit text masses can replace matching or likely-misclassified vision labels.",
            ],
        )
        return AgentResponse(
            source="ingredient_resolution",
            confidence=self._confidence(items),
            data=contract.model_dump(),
            error=None if items else "No resolvable ingredients were available.",
        )

    def _extract_outputs(self, input_data: Any) -> dict[str, AgentResponse | None]:
        if isinstance(input_data, AgentResponse):
            return self._extract_outputs(input_data.data)
        if not isinstance(input_data, dict):
            return {"vision": None, "text_nutrition": None}
        if "orchestrator" in input_data:
            orchestrator = self._coerce_response(input_data.get("orchestrator"))
            if orchestrator is not None:
                return self._extract_outputs(orchestrator.data)
        if input_data.get("contract_version") == "orchestrator.v1":
            nested = input_data.get("outputs") or {}
            return {
                "vision": self._coerce_response(nested.get("vision")),
                "text_nutrition": self._coerce_response(nested.get("text_nutrition")),
            }
        return {
            "vision": self._coerce_response(input_data.get("vision")),
            "text_nutrition": self._coerce_response(input_data.get("text_nutrition")),
        }

    def _coerce_response(self, value: Any) -> AgentResponse | None:
        if value is None:
            return None
        if isinstance(value, AgentResponse):
            return value
        if isinstance(value, dict):
            return AgentResponse.model_validate(value)
        return None

    def _resolve(
        self,
        vision_data: dict[str, Any],
        text_nutrition_data: dict[str, Any],
        decision_data: dict[str, Any],
        semantic_decisions: list[dict[str, Any]] | None = None,
    ) -> tuple[list[ResolvedIngredient], dict[str, str], list[str], list[str], list[str], list[str]]:
        vision_items = list(vision_data.get("ingredients") or [])
        text_items = list(text_nutrition_data.get("items") or [])
        explicit_text = [item for item in text_items if item.get("mass_source") == "explicit_text"]
        remaining_text = [item for item in text_items if item.get("mass_source") != "explicit_text"]
        replacements: dict[str, str] = {}
        assumptions: list[str] = []
        warnings: list[str] = []
        resolved: list[ResolvedIngredient] = []
        used_vision: set[int] = set()
        used_explicit_text: set[int] = set()
        used_remaining_text: set[int] = set()

        corrected = {
            self._canonical(str(key)): self._canonical(str(value))
            for key, value in (decision_data.get("corrected_food_names") or {}).items()
        }

        self._apply_semantic_decisions(
            decisions=semantic_decisions or [],
            vision_items=vision_items,
            explicit_text=explicit_text,
            remaining_text=remaining_text,
            resolved=resolved,
            replacements=replacements,
            assumptions=assumptions,
            warnings=warnings,
            used_vision=used_vision,
            used_explicit_text=used_explicit_text,
            used_remaining_text=used_remaining_text,
        )

        for text_index, text_item in enumerate(explicit_text):
            if text_index in used_explicit_text:
                continue
            text_canonical = self._canonical(str(text_item.get("lookup_name") or text_item.get("name")))
            match_index = self._find_vision_match(
                text_canonical=text_canonical,
                vision_items=vision_items,
                used_vision=used_vision,
                corrected=corrected,
            )
            if match_index is None:
                match_index = self._fallback_replacement_index(
                    explicit_text=explicit_text,
                    vision_items=vision_items,
                    used_vision=used_vision,
                )

            if match_index is not None:
                used_vision.add(match_index)
                replacements[str(vision_items[match_index].get("name"))] = str(text_item.get("name"))
                assumptions.append(
                    f"Text ingredient '{text_item.get('name')}' replaced vision label '{vision_items[match_index].get('name')}'."
                )
                resolved.append(
                    self._resolved_from_text(
                        text_item,
                        vision_item=vision_items[match_index],
                        action="replace",
                    )
                )
            else:
                assumptions.append(
                    f"Text ingredient '{text_item.get('name')}' was added as an explicit extra ingredient."
                )
                resolved.append(self._resolved_from_text(text_item, vision_item=None, action="add"))
            used_explicit_text.add(text_index)

        non_explicit_links = self._link_non_explicit_text(
            remaining_text=[
                item
                for index, item in enumerate(remaining_text)
                if index not in used_remaining_text
            ],
            vision_items=vision_items,
            used_vision=used_vision,
        )
        for text_item, vision_index in non_explicit_links:
            used_vision.add(vision_index)
            replacements[str(vision_items[vision_index].get("name"))] = str(text_item.get("name"))
            remaining_index = self._text_index(remaining_text, text_item)
            if remaining_index is not None:
                used_remaining_text.add(remaining_index)
            assumptions.append(
                f"Text ingredient '{text_item.get('name')}' relabeled vision item '{vision_items[vision_index].get('name')}' while keeping vision mass/nutrition."
            )
            resolved.append(
                self._resolved_from_vision_relabel(
                    vision_item=vision_items[vision_index],
                    text_item=text_item,
                )
            )

        remaining_anchor = self._remaining_anchor(
            vision_totals=vision_data.get("totals") or {},
            used_text_items=[
                explicit_text[index]
                for index in used_explicit_text
                if index < len(explicit_text)
            ],
            warnings=warnings,
        )
        remaining_vision = [
            item for index, item in enumerate(vision_items) if index not in used_vision
        ]
        if remaining_vision:
            if used_explicit_text:
                for item in remaining_vision:
                    resolved.append(self._resolved_from_vision(item))
            else:
                scaled_remaining = self._scaled_vision_items(remaining_vision, remaining_anchor)
                for item in scaled_remaining:
                    resolved.append(self._resolved_from_vision(item))

        if not resolved:
            for item in remaining_text:
                resolved.append(self._resolved_from_text(item, vision_item=None, action="add"))

        unmatched_vision = [
            str(item.get("name"))
            for index, item in enumerate(vision_items)
            if index not in used_vision and not remaining_vision
        ]
        unmatched_text = [
            str(item.get("name"))
            for index, item in enumerate(explicit_text)
            if index not in used_explicit_text
        ]
        return resolved, replacements, unmatched_vision, unmatched_text, assumptions, warnings

    async def _llm_semantic_decisions(
        self,
        vision_data: dict[str, Any],
        text_nutrition_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError(
                "langchain, langchain-core, and langchain-ollama are required for semantic ingredient resolution."
            ) from exc

        context = {
            "vision_items": [
                {
                    "name": item.get("name"),
                    "confidence": item.get("confidence"),
                    "mass_g": (item.get("nutrition") or {}).get("mass_g"),
                    "calories_kcal": (item.get("nutrition") or {}).get("calories_kcal"),
                }
                for item in vision_data.get("ingredients") or []
            ],
            "text_items": [
                {
                    "name": item.get("name"),
                    "canonical_name": item.get("lookup_name"),
                    "mass_g": item.get("mass_g"),
                    "mass_source": item.get("mass_source"),
                    "confidence": item.get("confidence"),
                    "canonicalization_source": item.get("canonicalization_source"),
                }
                for item in text_nutrition_data.get("items") or []
            ],
        }
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You resolve semantic relationships between food items detected by vision and food items "
                    "provided by user text. Vision labels can be visually plausible but wrong. User text can be "
                    "partial and may correct only one visible item. Return JSON only. Do not calculate nutrition, "
                    "calories, macros, or grams. Use only names from the provided JSON for text_food and vision_food. "
                    "Allowed actions: replace, match, add, preserve, ignore. "
                    "Use replace when text likely corrects a vision misclassification. Use match when they are the "
                    "same food. Use add when a text food is extra and not represented by vision. Use preserve for "
                    "vision foods not contradicted by text. Use ignore for malformed/non-food text. "
                    "If text has explicit mass and plausibly refers to a vision item, prefer replace/match with "
                    "mass_policy use_text_mass and nutrition_policy use_text_nutrition. If text has no explicit mass "
                    "but corrects a vision item, use mass_policy use_vision_mass and nutrition_policy use_vision_nutrition.",
                ),
                (
                    "human",
                    "Evidence JSON:\n{context_json}\n\n"
                    "Return JSON exactly in this shape: "
                    "{{\"decisions\":[{{\"action\":\"replace|match|add|preserve|ignore\","
                    "\"text_food\":\"... or null\",\"vision_food\":\"... or null\","
                    "\"canonical_food\":\"...\",\"mass_policy\":\"use_text_mass|use_vision_mass|none\","
                    "\"nutrition_policy\":\"use_text_nutrition|use_vision_nutrition|none\","
                    "\"confidence\":0.0,\"reason\":\"...\"}}]}}",
                ),
            ]
        )
        llm = ChatOllama(
            model=self.model,
            temperature=0.0,
            reasoning=False,
            format="json",
            num_predict=768,
            async_client_kwargs={"timeout": self.timeout_seconds},
        )
        result = await (prompt | llm).ainvoke(
            {"context_json": json.dumps(context, ensure_ascii=False, sort_keys=True)}
        )
        content = result.content if hasattr(result, "content") else result
        if isinstance(content, list):
            content = "".join(str(part) for part in content)
        data = json.loads(str(content))
        raw_decisions = data.get("decisions") if isinstance(data, dict) else None
        if not isinstance(raw_decisions, list):
            raise ValueError("LLM ingredient resolution returned JSON without a decisions list.")
        return [
            decision
            for decision in (self._normalize_semantic_decision(item) for item in raw_decisions)
            if decision is not None
        ]

    def _normalize_semantic_decision(self, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        action = str(value.get("action") or "").strip().lower()
        if action not in {"replace", "match", "add", "preserve", "ignore"}:
            return None
        mass_policy = str(value.get("mass_policy") or "none").strip().lower()
        if mass_policy not in {"use_text_mass", "use_vision_mass", "none"}:
            mass_policy = "none"
        nutrition_policy = str(value.get("nutrition_policy") or "none").strip().lower()
        if nutrition_policy not in {"use_text_nutrition", "use_vision_nutrition", "none"}:
            nutrition_policy = "none"
        try:
            confidence = max(min(float(value.get("confidence", 0.65)), 1.0), 0.0)
        except (TypeError, ValueError):
            confidence = 0.65
        return {
            "action": action,
            "text_food": self._optional_text(value.get("text_food")),
            "vision_food": self._optional_text(value.get("vision_food")),
            "canonical_food": self._optional_text(value.get("canonical_food")),
            "mass_policy": mass_policy,
            "nutrition_policy": nutrition_policy,
            "confidence": confidence,
            "reason": str(value.get("reason") or "").strip(),
        }

    def _apply_semantic_decisions(
        self,
        decisions: list[dict[str, Any]],
        vision_items: list[dict[str, Any]],
        explicit_text: list[dict[str, Any]],
        remaining_text: list[dict[str, Any]],
        resolved: list[ResolvedIngredient],
        replacements: dict[str, str],
        assumptions: list[str],
        warnings: list[str],
        used_vision: set[int],
        used_explicit_text: set[int],
        used_remaining_text: set[int],
    ) -> None:
        for decision in decisions:
            action = str(decision.get("action"))
            reason = str(decision.get("reason") or "").strip()
            confidence = float(decision.get("confidence") or 0.65)
            vision_index = self._vision_index_for_decision(decision, vision_items, used_vision)
            text_location = self._text_location_for_decision(
                decision=decision,
                explicit_text=explicit_text,
                remaining_text=remaining_text,
                used_explicit_text=used_explicit_text,
                used_remaining_text=used_remaining_text,
            )

            if action in {"replace", "match"}:
                if vision_index is None or text_location is None:
                    warnings.append(f"LLM {action} decision ignored because referenced items were not available.")
                    continue
                text_kind, text_index, text_item = text_location
                vision_item = vision_items[vision_index]
                used_vision.add(vision_index)
                if text_kind == "explicit":
                    used_explicit_text.add(text_index)
                    item = self._resolved_from_text(
                        text_item,
                        vision_item=vision_item,
                        action="replace" if action == "replace" else "match",
                    )
                else:
                    used_remaining_text.add(text_index)
                    item = self._resolved_from_vision_relabel(
                        vision_item=vision_item,
                        text_item=text_item,
                    )
                item.confidence = min(max((item.confidence + confidence) / 2, 0.0), 1.0)
                if reason:
                    item.notes.append(reason)
                resolved.append(item)
                replacements[str(vision_item.get("name"))] = str(text_item.get("name"))
                assumptions.append(
                    f"LLM semantic resolver {action}d text ingredient '{text_item.get('name')}' with vision item '{vision_item.get('name')}'."
                )
                continue

            if action == "add":
                if text_location is None:
                    warnings.append("LLM add decision ignored because referenced text item was not available.")
                    continue
                text_kind, text_index, text_item = text_location
                if text_item.get("mass_source") != "explicit_text":
                    if text_kind == "remaining":
                        used_remaining_text.add(text_index)
                    assumptions.append(
                        f"LLM marked text ingredient '{text_item.get('name')}' as extra, but it was not quantified; it was not added to numeric totals."
                    )
                    continue
                used_explicit_text.add(text_index)
                item = self._resolved_from_text(text_item, vision_item=None, action="add")
                item.confidence = min(max((item.confidence + confidence) / 2, 0.0), 1.0)
                if reason:
                    item.notes.append(reason)
                resolved.append(item)
                assumptions.append(
                    f"LLM semantic resolver added explicit text ingredient '{text_item.get('name')}'."
                )
                continue

            if action == "preserve":
                if vision_index is None:
                    continue
                used_vision.add(vision_index)
                item = self._resolved_from_vision(vision_items[vision_index])
                item.confidence = min(max((item.confidence + confidence) / 2, 0.0), 1.0)
                if reason:
                    item.notes.append(reason)
                resolved.append(item)
                continue

            if action == "ignore" and text_location is not None:
                text_kind, text_index, text_item = text_location
                if text_kind == "explicit":
                    used_explicit_text.add(text_index)
                else:
                    used_remaining_text.add(text_index)
                assumptions.append(
                    f"LLM semantic resolver ignored text ingredient '{text_item.get('name')}'."
                )

    def _vision_index_for_decision(
        self,
        decision: dict[str, Any],
        vision_items: list[dict[str, Any]],
        used_vision: set[int],
    ) -> int | None:
        names = [
            decision.get("vision_food"),
            decision.get("canonical_food"),
        ]
        for name in names:
            if not name:
                continue
            normalized = self._canonical(str(name))
            for index, item in enumerate(vision_items):
                if index in used_vision:
                    continue
                item_name = self._canonical(str(item.get("name") or ""))
                if item_name == normalized or self._normalize(str(item.get("name") or "")) == self._normalize(str(name)):
                    return index
        return None

    def _text_location_for_decision(
        self,
        decision: dict[str, Any],
        explicit_text: list[dict[str, Any]],
        remaining_text: list[dict[str, Any]],
        used_explicit_text: set[int],
        used_remaining_text: set[int],
    ) -> tuple[str, int, dict[str, Any]] | None:
        names = [
            decision.get("text_food"),
            decision.get("canonical_food"),
        ]
        for name in names:
            if not name:
                continue
            normalized = self._canonical(str(name))
            for index, item in enumerate(explicit_text):
                if index in used_explicit_text:
                    continue
                if self._text_item_matches(item, normalized, str(name)):
                    return "explicit", index, item
            for index, item in enumerate(remaining_text):
                if index in used_remaining_text:
                    continue
                if self._text_item_matches(item, normalized, str(name)):
                    return "remaining", index, item
        return None

    def _text_item_matches(self, item: dict[str, Any], canonical_name: str, raw_name: str) -> bool:
        item_names = {
            self._canonical(str(item.get("name") or "")),
            self._canonical(str(item.get("lookup_name") or "")),
            self._normalize(str(item.get("name") or "")),
            self._normalize(str(item.get("lookup_name") or "")),
        }
        return canonical_name in item_names or self._normalize(raw_name) in item_names

    def _text_index(self, items: list[dict[str, Any]], target: dict[str, Any]) -> int | None:
        for index, item in enumerate(items):
            if item is target:
                return index
        return None

    def _optional_text(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in {"none", "null", "n/a"}:
            return None
        return text

    def _find_vision_match(
        self,
        text_canonical: str,
        vision_items: list[dict[str, Any]],
        used_vision: set[int],
        corrected: dict[str, str],
    ) -> int | None:
        for index, item in enumerate(vision_items):
            if index in used_vision:
                continue
            vision_canonical = self._canonical(str(item.get("name")))
            if vision_canonical == text_canonical:
                return index
            if corrected.get(vision_canonical) == text_canonical:
                return index
        return None

    def _link_non_explicit_text(
        self,
        remaining_text: list[dict[str, Any]],
        vision_items: list[dict[str, Any]],
        used_vision: set[int],
    ) -> list[tuple[dict[str, Any], int]]:
        if not remaining_text:
            return []

        available = [index for index in range(len(vision_items)) if index not in used_vision]
        links: list[tuple[dict[str, Any], int]] = []
        used_available: set[int] = set()

        for text_item in remaining_text:
            text_canonical = self._canonical(str(text_item.get("lookup_name") or text_item.get("name")))
            match = self._find_vision_match(
                text_canonical=text_canonical,
                vision_items=vision_items,
                used_vision=used_vision | used_available,
                corrected={},
            )
            if match is not None:
                links.append((text_item, match))
                used_available.add(match)

        if links:
            return links

        if len(remaining_text) == len(available):
            return [
                (text_item, vision_index)
                for text_item, vision_index in zip(remaining_text, available)
            ]

        if len(remaining_text) == 1:
            text_canonical = self._canonical(
                str(remaining_text[0].get("lookup_name") or remaining_text[0].get("name"))
            )
            text_group = self._food_group(text_canonical)
            if text_group is not None:
                grouped = [
                    index
                    for index in available
                    if self._food_group(self._canonical(str(vision_items[index].get("name")))) == text_group
                ]
                if len(grouped) == 1:
                    return [(remaining_text[0], grouped[0])]

        return []

    def _fallback_replacement_index(
        self,
        explicit_text: list[dict[str, Any]],
        vision_items: list[dict[str, Any]],
        used_vision: set[int],
    ) -> int | None:
        available = [index for index in range(len(vision_items)) if index not in used_vision]
        if len(explicit_text) == 1:
            text_canonical = self._canonical(
                str(explicit_text[0].get("lookup_name") or explicit_text[0].get("name"))
            )
            text_group = self._food_group(text_canonical)
            if text_group is not None:
                grouped = [
                    index
                    for index in available
                    if self._food_group(self._canonical(str(vision_items[index].get("name")))) == text_group
                ]
                if len(grouped) == 1:
                    return grouped[0]

        if len(explicit_text) == 1 and len(available) == 1:
            return available[0]
        if len(explicit_text) == len(vision_items):
            return available[0] if available else None
        return None

    def _resolved_from_text(
        self,
        text_item: dict[str, Any],
        vision_item: dict[str, Any] | None,
        action: str,
    ) -> ResolvedIngredient:
        name = str(text_item.get("name") or text_item.get("lookup_name") or "unknown food")
        canonical = self._canonical(str(text_item.get("lookup_name") or name))
        nutrition = NutritionEstimate.model_validate(text_item.get("nutrition") or {})
        links = [
            IngredientSourceLink(
                source="text_nutrition",
                name=name,
                confidence=text_item.get("confidence"),
                mass_g=text_item.get("mass_g"),
                action=action,
            )
        ]
        notes = [f"Resolved from explicit text nutrition using {text_item.get('mass_source')} mass."]
        if vision_item is not None:
            links.append(
                IngredientSourceLink(
                    source="vision",
                    name=str(vision_item.get("name")),
                    confidence=vision_item.get("confidence"),
                    mass_g=(vision_item.get("nutrition") or {}).get("mass_g"),
                    action="replace" if action == "replace" else "match",
                )
            )
            notes.append(f"Replaced vision label '{vision_item.get('name')}'.")
        return ResolvedIngredient(
            display_name=name,
            canonical_name=canonical,
            source="mixed" if vision_item is not None else "text_nutrition",
            mass_g=nutrition.mass_g,
            mass_source="explicit_text" if text_item.get("mass_source") == "explicit_text" else "default_serving",
            nutrition=nutrition,
            confidence=float(text_item.get("confidence") or 0.7),
            links=links,
            notes=notes,
        )

    def _resolved_from_vision(self, vision_item: dict[str, Any]) -> ResolvedIngredient:
        name = str(vision_item.get("name") or "unknown food")
        nutrition = NutritionEstimate.model_validate(vision_item.get("nutrition") or {})
        return ResolvedIngredient(
            display_name=name,
            canonical_name=self._canonical(name),
            source="vision",
            mass_g=nutrition.mass_g,
            mass_source="vision_remaining",
            nutrition=nutrition,
            confidence=float(vision_item.get("confidence") or 0.65),
            links=[
                IngredientSourceLink(
                    source="vision",
                    name=name,
                    confidence=vision_item.get("confidence"),
                    mass_g=nutrition.mass_g,
                    action="preserve",
                )
            ],
            notes=["Preserved visible vision ingredient after text replacements."],
        )

    def _resolved_from_vision_relabel(
        self,
        vision_item: dict[str, Any],
        text_item: dict[str, Any],
    ) -> ResolvedIngredient:
        display_name = str(text_item.get("name") or text_item.get("lookup_name") or vision_item.get("name"))
        canonical_name = self._canonical(str(text_item.get("lookup_name") or display_name))
        nutrition = NutritionEstimate.model_validate(vision_item.get("nutrition") or {})
        return ResolvedIngredient(
            display_name=display_name,
            canonical_name=canonical_name,
            source="mixed",
            mass_g=nutrition.mass_g,
            mass_source="vision",
            nutrition=nutrition,
            confidence=min(float(text_item.get("confidence") or 0.65), float(vision_item.get("confidence") or 0.65)),
            links=[
                IngredientSourceLink(
                    source="text_nutrition",
                    name=display_name,
                    confidence=text_item.get("confidence"),
                    mass_g=text_item.get("mass_g"),
                    action="match",
                ),
                IngredientSourceLink(
                    source="vision",
                    name=str(vision_item.get("name")),
                    confidence=vision_item.get("confidence"),
                    mass_g=nutrition.mass_g,
                    action="replace",
                ),
            ],
            notes=[
                f"Display identity came from text; mass and nutrition remain from vision because text had no explicit grams."
            ],
        )

    def _remaining_anchor(
        self,
        vision_totals: dict[str, Any],
        used_text_items: list[dict[str, Any]],
        warnings: list[str],
    ) -> NutritionEstimate:
        used = self._sum_text_items(used_text_items)
        if used.mass_g and vision_totals.get("mass_g") is not None and used.mass_g > float(vision_totals["mass_g"]):
            warnings.append("Explicit text mass exceeds vision mass; no remaining vision mass was allocated.")
        return NutritionEstimate(
            calories_kcal=self._subtract_floor(vision_totals.get("calories_kcal"), used.calories_kcal),
            mass_g=self._subtract_floor(vision_totals.get("mass_g"), used.mass_g),
            protein_g=self._subtract_floor(vision_totals.get("protein_g"), used.protein_g),
            carbs_g=self._subtract_floor(vision_totals.get("carbs_g"), used.carbs_g),
            fat_g=self._subtract_floor(vision_totals.get("fat_g"), used.fat_g),
        )

    def _scaled_vision_items(
        self,
        vision_items: list[dict[str, Any]],
        anchor: NutritionEstimate,
    ) -> list[dict[str, Any]]:
        total = self._sum_vision_items(vision_items)
        calorie_scale = anchor.calories_kcal / total.calories_kcal if anchor.calories_kcal is not None and total.calories_kcal else 1.0
        mass_scale = anchor.mass_g / total.mass_g if anchor.mass_g is not None and total.mass_g else 1.0
        scaled = []
        for item in vision_items:
            nutrition = item.get("nutrition") or {}
            scaled_item = dict(item)
            scaled_item["nutrition"] = {
                "calories_kcal": self._scale(nutrition.get("calories_kcal"), calorie_scale),
                "mass_g": self._scale(nutrition.get("mass_g"), mass_scale),
                "protein_g": self._scale(nutrition.get("protein_g"), calorie_scale),
                "carbs_g": self._scale(nutrition.get("carbs_g"), calorie_scale),
                "fat_g": self._scale(nutrition.get("fat_g"), calorie_scale),
            }
            scaled.append(scaled_item)
        return scaled

    def _sum_text_items(self, items: list[dict[str, Any]]) -> NutritionEstimate:
        return self._sum_nutrition(item.get("nutrition") or {} for item in items)

    def _sum_vision_items(self, items: list[dict[str, Any]]) -> NutritionEstimate:
        return self._sum_nutrition(item.get("nutrition") or {} for item in items)

    def _sum_items(self, items: list[ResolvedIngredient]) -> NutritionEstimate:
        return NutritionEstimate(
            calories_kcal=self._sum_optional(item.nutrition.calories_kcal for item in items),
            mass_g=self._sum_optional(item.nutrition.mass_g for item in items),
            protein_g=self._sum_optional(item.nutrition.protein_g for item in items),
            carbs_g=self._sum_optional(item.nutrition.carbs_g for item in items),
            fat_g=self._sum_optional(item.nutrition.fat_g for item in items),
        )

    def _sum_nutrition(self, values: Any) -> NutritionEstimate:
        values = list(values)
        return NutritionEstimate(
            calories_kcal=self._sum_optional(value.get("calories_kcal") for value in values),
            mass_g=self._sum_optional(value.get("mass_g") for value in values),
            protein_g=self._sum_optional(value.get("protein_g") for value in values),
            carbs_g=self._sum_optional(value.get("carbs_g") for value in values),
            fat_g=self._sum_optional(value.get("fat_g") for value in values),
        )

    def _sum_optional(self, values: Any) -> float | None:
        present = [float(value) for value in values if value is not None]
        return sum(present) if present else None

    def _subtract_floor(self, left: Any, right: Any) -> float | None:
        if left is None:
            return None
        return max(float(left) - float(right or 0.0), 0.0)

    def _scale(self, value: Any, factor: float) -> float | None:
        if value is None:
            return None
        return float(value) * factor

    def _confidence(self, items: list[ResolvedIngredient]) -> float:
        if not items:
            return 0.0
        return sum(item.confidence for item in items) / len(items)

    def _canonical(self, value: str) -> str:
        normalized = self._normalize(value)
        return self.aliases.get(normalized, normalized)

    def _food_group(self, canonical: str) -> str | None:
        return self.FOOD_GROUPS.get(canonical)

    def _normalize(self, value: str) -> str:
        return re.sub(r"[^a-z0-9çğıöşüı]+", " ", value.lower()).strip()

    def _load_aliases(self, path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {self._normalize(key): self._normalize(str(value)) for key, value in raw.items()}
