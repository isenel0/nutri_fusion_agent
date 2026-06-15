"""Deterministic fusion agent for multimodal meal analysis."""

from __future__ import annotations

import re
from typing import Any

from agents.base_agent import BaseAgent
from agents.vision.nutrition_lookup import NutritionLookup
from schemas import (
    AgentResponse,
    FusionDecision,
    MealAnalysisResult,
    MealItemResult,
    NutritionEstimate,
)


class FusionAgent(BaseAgent):
    """Combines vision, text, and barcode evidence with explicit rules.

    This layer performs deterministic math only. A future LLM layer should
    produce or adjust a fusion decision, then this agent should still compute
    the final nutrition numbers.
    """

    def __init__(self) -> None:
        self.nutrition_lookup = NutritionLookup()

    async def process(self, input_data: Any) -> AgentResponse:
        decision_response = self._extract_decision_response(input_data)
        if decision_response is not None and decision_response.error:
            return AgentResponse(
                source="fusion",
                confidence=0.0,
                data={},
                error=(
                    "Qwen fusion decision is required, but the decision agent failed: "
                    f"{decision_response.error}"
                ),
            )

        outputs = self._extract_outputs(input_data)
        llm_decision = self._extract_decision(input_data)
        depth = outputs.get("depth")
        vision = outputs.get("vision")
        text = outputs.get("text")
        text_nutrition = outputs.get("text_nutrition")
        ingredient_resolution = outputs.get("ingredient_resolution")
        barcode = outputs.get("barcode")

        vision_data = vision.data if vision else {}
        text_data = text.data if text else {}
        text_nutrition_data = text_nutrition.data if text_nutrition else {}
        ingredient_resolution_data = ingredient_resolution.data if ingredient_resolution else {}
        barcode_data = barcode.data if barcode else {}

        portion_multiplier = self._portion_multiplier(text_data, llm_decision)
        mass_g, mass_source = self._resolve_mass_g(vision_data, text_data)
        applied_rules: list[str] = []
        conflicts: list[str] = []
        assumptions: list[str] = []
        warnings: list[str] = []

        barcode_nutrition = barcode_data.get("nutrition_per_100g") or {}
        has_barcode_macros = self._has_any_macro(barcode_nutrition)
        vision_totals = vision_data.get("totals") or {}
        has_vision_totals = self._has_any_macro(vision_totals)
        text_nutrition_totals = text_nutrition_data.get("totals") or {}
        has_text_nutrition = self._has_any_macro(text_nutrition_totals)
        has_explicit_text_mass = self._has_explicit_text_mass(text_nutrition_data)
        ingredient_resolution_totals = ingredient_resolution_data.get("totals") or {}

        primary_source = self._primary_source(
            llm_decision=llm_decision,
            has_barcode_macros=has_barcode_macros,
            has_text_nutrition=has_text_nutrition,
            has_explicit_text_mass=has_explicit_text_mass,
            has_vision_totals=has_vision_totals,
            warnings=warnings,
        )
        if primary_source != "barcode" and self._has_any_macro(ingredient_resolution_totals):
            primary_source = "ingredient_resolution"
        if (
            primary_source == "text_nutrition"
            and has_vision_totals
            and vision_data.get("ingredients")
            and not self._explicit_text_covers_vision(text_nutrition_data, vision_data, llm_decision)
        ):
            primary_source = "vision"
            warnings.append(
                "Text has explicit quantities for only part of the meal; vision was kept for missing visible ingredients."
            )

        if primary_source == "barcode":
            applied_rules.append("barcode_nutrition_overrides_generic_estimates")
            if mass_g is None:
                mass_g = 100.0
                mass_source = "assumed_100g"
                assumptions.append(
                    "No serving mass was available, so barcode nutrition was reported for an assumed 100g serving."
                )
            final_macros = self._scale_per_100g(
                barcode_nutrition,
                mass_g * portion_multiplier,
            )
        elif primary_source == "vision":
            applied_rules.append("vision_late_fusion_totals_used")
            final_macros = self._scale_totals(vision_totals, portion_multiplier)
        elif primary_source == "text_nutrition":
            applied_rules.append("text_nutrition_with_explicit_or_assumed_mass_used")
            final_macros = self._scale_totals(text_nutrition_totals, portion_multiplier)
        elif primary_source == "ingredient_resolution":
            applied_rules.append("ingredient_resolution_totals_used")
            final_macros = self._scale_totals(ingredient_resolution_totals, portion_multiplier)
        else:
            warnings.append("No numeric nutrition estimate was available from barcode, vision, text nutrition, or ingredient resolution.")
            final_macros = NutritionEstimate()

        if portion_multiplier != 1.0:
            applied_rules.append("text_portion_multiplier_applied")

        self._detect_conflicts(vision_data, text_data, barcode_data, conflicts)
        if llm_decision:
            applied_rules.extend(llm_decision.get("applied_rules", []))
            conflicts.extend(llm_decision.get("conflicts", []))
            assumptions.extend(llm_decision.get("assumptions", []))

        name_overrides = self._name_overrides(
            vision_data,
            text_data,
            barcode_data,
            llm_decision,
            ingredient_resolution_data if ingredient_resolution is not None else None,
        )
        if name_overrides:
            applied_rules.append("ingredient_names_reconciled_from_text_and_llm")
            assumptions.append(
                "Ingredient labels were reconciled across agents; nutrition values still come from the selected numeric source."
            )

        text_foods = self._text_foods(text_data)
        hidden_ingredients = (llm_decision or {}).get("hidden_ingredients", [])
        hidden_tokens = {
            self._normalize_food_name(str(ingredient))
            for ingredient in hidden_ingredients
        }
        calculation_text_foods = [
            food
            for food in text_foods
            if self._normalize_food_name(food) not in hidden_tokens
        ]
        text_guided_items = False
        mixed_text_vision_items = False
        barcode_match_items = (
            ingredient_resolution_data.get("items")
            or self._resolution_items_from_text_nutrition(text_nutrition_data)
        )
        barcode_target_map = self._barcode_target_map(
            barcode_data=barcode_data,
            resolution_items=barcode_match_items,
        )
        if (
            barcode_target_map
            and barcode_match_items
        ):
            primary_source = "barcode"
            items, used_explicit_barcode_mass = self._build_multi_barcode_guided_items(
                barcode_data=barcode_data,
                resolution_items=barcode_match_items,
                barcode_target_map=barcode_target_map,
                fallback_mass_g=mass_g,
                portion_multiplier=portion_multiplier,
                confidence=self._confidence(outputs, conflicts=conflicts),
            )
            final_macros = self._sum_item_nutrition(items)
            if used_explicit_barcode_mass:
                mass_source = "text"
                applied_rules.append("barcode_nutrition_scaled_by_explicit_text_mass")
            applied_rules.append("barcode_nutrition_applied_to_matched_ingredient")
            if ingredient_resolution_data.get("items"):
                assumptions.extend(ingredient_resolution_data.get("assumptions") or [])
                warnings.extend(ingredient_resolution_data.get("warnings") or [])
        elif primary_source == "barcode" and has_barcode_macros:
            items, used_explicit_barcode_mass = self._build_barcode_guided_items(
                barcode_data=barcode_data,
                resolution_items=barcode_match_items,
                barcode_nutrition=barcode_nutrition,
                fallback_mass_g=mass_g,
                portion_multiplier=portion_multiplier,
                name_overrides=name_overrides,
                confidence=self._confidence(outputs, conflicts=conflicts),
            )
            final_macros = self._sum_item_nutrition(items)
            if used_explicit_barcode_mass:
                mass_source = "text"
                applied_rules.append("barcode_nutrition_scaled_by_explicit_text_mass")
        elif primary_source != "barcode" and ingredient_resolution_data.get("items"):
            items = self._items_from_resolution(
                ingredient_resolution_data=ingredient_resolution_data,
                confidence=self._confidence(outputs, conflicts=conflicts),
            )
            final_macros = self._sum_item_nutrition(items)
            applied_rules.append("ingredient_resolution_used")
            if self._resolution_uses_only_explicit_text(ingredient_resolution_data):
                mass_source = "text"
                applied_rules.append("complete_explicit_text_mass_used")
            assumptions.extend(ingredient_resolution_data.get("assumptions") or [])
            warnings.extend(ingredient_resolution_data.get("warnings") or [])
            if ingredient_resolution_data.get("replacements"):
                name_overrides = {
                    **name_overrides,
                    **{
                        str(key): str(value)
                        for key, value in ingredient_resolution_data.get("replacements", {}).items()
                    },
                }
        elif primary_source == "vision" and self._has_any_explicit_text_mass(text_nutrition_data):
            mixed_text_vision_items = True
            items = self._build_mixed_text_vision_items(
                vision_data=vision_data,
                text_nutrition_data=text_nutrition_data,
                portion_multiplier=portion_multiplier,
                name_overrides=name_overrides,
                anchor=final_macros,
                confidence=self._confidence(outputs, conflicts=conflicts),
                warnings=warnings,
            )
            final_macros = self._sum_item_nutrition(items)
            applied_rules.append("partial_text_quantities_combined_with_remaining_vision_items")
            assumptions.append(
                "Explicit text quantities were used for matching ingredients; vision supplied remaining visible ingredients and remaining mass/calorie budget."
            )
        elif self._should_use_text_guided_vision_items(
            primary_source=primary_source,
                vision_data=vision_data,
                text_nutrition_data=text_nutrition_data,
                text_foods=calculation_text_foods,
                hidden_ingredients=hidden_ingredients,
                name_overrides=name_overrides,
            llm_decision=llm_decision,
        ):
            text_guided_items = True
            items = self._build_text_guided_vision_items(
                text_foods=calculation_text_foods,
                final_macros=final_macros,
                text_nutrition_data=text_nutrition_data,
                confidence=self._confidence(
                    outputs,
                    conflicts=conflicts,
                    text_guided_items=True,
                ),
            )
            final_macros = self._sum_item_nutrition(items)
            hidden_ingredients = [
                ingredient
                for ingredient in hidden_ingredients
                if self._normalize_food_name(str(ingredient))
                not in {self._normalize_food_name(food) for food in calculation_text_foods}
            ]
            applied_rules.append("text_foods_override_conflicting_vision_ingredient_labels")
            assumptions.append(
                "Vision supplied the total mass and calorie anchor, while explicit text foods supplied the calculation ingredient list."
            )
        else:
            items = self._build_items(
                primary_source=primary_source,
                vision_data=vision_data,
                text_nutrition_data=text_nutrition_data,
                barcode_data=barcode_data,
                final_macros=final_macros,
                mass_g=mass_g,
                portion_multiplier=portion_multiplier,
                name_overrides=name_overrides,
                confidence=self._confidence(outputs, conflicts=conflicts),
            )

        if hidden_ingredients:
            assumptions.append(
                "LLM detected hidden ingredients not yet included in deterministic macro math: "
                + ", ".join(str(item) for item in hidden_ingredients)
            )

        confidence = self._confidence(
            outputs,
            conflicts=conflicts,
            warnings=warnings,
            text_guided_items=text_guided_items or mixed_text_vision_items,
        )

        decision = FusionDecision(
            primary_nutrition_source=primary_source,
            mass_source=mass_source,
            portion_multiplier=portion_multiplier,
            corrected_food_names=name_overrides,
            hidden_ingredients=hidden_ingredients,
            applied_rules=applied_rules,
            conflicts=conflicts,
            assumptions=assumptions,
            explanation=(llm_decision or {}).get("explanation"),
        )
        result = MealAnalysisResult(
            meal_name=self._meal_name(text_data, vision_data, barcode_data),
            final_macros=final_macros,
            calculation_ingredients=items,
            items=items,
            fusion_decision=decision,
            inputs_used={
                "vision": vision is not None,
                "depth": depth is not None,
                "text": text is not None,
                "text_nutrition": text_nutrition is not None,
                "ingredient_resolution": ingredient_resolution is not None,
                "barcode": barcode is not None,
            },
            confidence=confidence,
            reasoning_summary=self._reasoning_summary(decision, warnings),
            warnings=warnings,
            agent_outputs=outputs,
        )

        return AgentResponse(
            source="fusion",
            confidence=result.confidence,
            data=result.model_dump(),
            error=None if self._has_any_macro(result.final_macros.model_dump()) else "No final numeric macros could be computed.",
        )

    def _extract_outputs(self, input_data: Any) -> dict[str, AgentResponse | None]:
        if isinstance(input_data, AgentResponse):
            return self._extract_outputs(input_data.data)

        if not isinstance(input_data, dict):
            return {"vision": None, "text": None, "barcode": None}

        if "orchestrator" in input_data:
            orchestrator = self._coerce_response(input_data.get("orchestrator"))
            if orchestrator is not None:
                return self._extract_outputs(orchestrator.data)

        if input_data.get("contract_version") == "orchestrator.v1":
            nested = input_data.get("outputs") or {}
            return {
                "vision": self._coerce_response(nested.get("vision")),
                "depth": self._coerce_response(nested.get("depth")),
                "text": self._coerce_response(nested.get("text")),
                "text_nutrition": self._coerce_response(nested.get("text_nutrition")),
                "ingredient_resolution": self._coerce_response(nested.get("ingredient_resolution")),
                "barcode": self._coerce_response(nested.get("barcode")),
            }

        return {
            "vision": self._coerce_response(input_data.get("vision")),
            "depth": self._coerce_response(input_data.get("depth")),
            "text": self._coerce_response(input_data.get("text")),
            "text_nutrition": self._coerce_response(input_data.get("text_nutrition")),
            "ingredient_resolution": self._coerce_response(input_data.get("ingredient_resolution")),
            "barcode": self._coerce_response(input_data.get("barcode")),
        }

    def _coerce_response(self, value: Any) -> AgentResponse | None:
        if value is None:
            return None
        if isinstance(value, AgentResponse):
            return value
        if isinstance(value, dict):
            return AgentResponse.model_validate(value)
        return None

    def _extract_decision(self, input_data: Any) -> dict[str, Any] | None:
        decision_response = self._extract_decision_response(input_data)
        if decision_response is not None and decision_response.data:
            return decision_response.data
        if isinstance(input_data, AgentResponse):
            return self._extract_decision(input_data.data)
        if not isinstance(input_data, dict):
            return None
        decision = input_data.get("fusion_decision")
        if isinstance(decision, dict):
            return decision
        return None

    def _extract_decision_response(self, input_data: Any) -> AgentResponse | None:
        if isinstance(input_data, AgentResponse):
            return self._extract_decision_response(input_data.data)
        if not isinstance(input_data, dict):
            return None
        decision_response = self._coerce_response(input_data.get("decision"))
        return decision_response

    def _portion_multiplier(
        self,
        text_data: dict[str, Any],
        llm_decision: dict[str, Any] | None,
    ) -> float:
        if llm_decision and llm_decision.get("portion_multiplier") is not None:
            try:
                return max(float(llm_decision["portion_multiplier"]), 0.0)
            except (TypeError, ValueError):
                pass

        parsed = text_data.get("parsed") or {}
        multiplier = parsed.get("portion_multiplier")
        if multiplier is None:
            return 1.0
        try:
            return max(float(multiplier), 0.0)
        except (TypeError, ValueError):
            return 1.0

    def _primary_source(
        self,
        llm_decision: dict[str, Any] | None,
        has_barcode_macros: bool,
        has_text_nutrition: bool,
        has_explicit_text_mass: bool,
        has_vision_totals: bool,
        warnings: list[str],
    ) -> str:
        requested = (llm_decision or {}).get("primary_nutrition_source")
        if requested == "barcode":
            if has_barcode_macros:
                return "barcode"
            warnings.append("LLM selected barcode, but barcode nutrition is unavailable.")
        if has_text_nutrition and has_explicit_text_mass and not has_vision_totals:
            if requested == "vision":
                warnings.append(
                    "LLM selected vision, but explicit text quantities are available; text nutrition was used."
                )
            return "text_nutrition"
        if requested == "vision":
            if has_vision_totals:
                return "vision"
            warnings.append("LLM selected vision, but vision totals are unavailable.")
        if requested == "text_nutrition":
            if has_text_nutrition:
                return "text_nutrition"
            warnings.append("LLM selected text nutrition, but text nutrition is unavailable.")
        if requested == "fallback":
            return "fallback"

        if has_barcode_macros:
            return "barcode"
        if has_text_nutrition and has_explicit_text_mass:
            return "text_nutrition"
        if has_vision_totals:
            return "vision"
        if has_text_nutrition:
            return "text_nutrition"
        return "fallback"

    def _resolve_mass_g(
        self,
        vision_data: dict[str, Any],
        text_data: dict[str, Any],
    ) -> tuple[float | None, str]:
        vision_mass = (vision_data.get("totals") or {}).get("mass_g")
        if vision_mass is not None:
            return float(vision_mass), "vision"

        text_mass = self._mass_from_text(text_data)
        if text_mass is not None:
            return text_mass, "text"

        return None, "unknown"

    def _mass_from_text(self, text_data: dict[str, Any]) -> float | None:
        parsed = text_data.get("parsed") or {}
        quantities = parsed.get("quantities") or []
        units = parsed.get("units") or []
        joined = " ".join([*quantities, *units]).lower()
        match = re.search(r"(\d+(?:\.\d+)?)\s*(g|gram|grams)", joined)
        if not match:
            return None
        return float(match.group(1))

    def _scale_per_100g(self, nutrition: dict[str, Any], grams: float) -> NutritionEstimate:
        factor = grams / 100.0
        return NutritionEstimate(
            calories_kcal=self._scale_value(nutrition.get("calories_kcal"), factor),
            mass_g=grams,
            protein_g=self._scale_value(nutrition.get("protein_g"), factor),
            carbs_g=self._scale_value(nutrition.get("carbs_g"), factor),
            fat_g=self._scale_value(nutrition.get("fat_g"), factor),
            fiber_g=self._scale_value(nutrition.get("fiber_g"), factor),
            sugar_g=self._scale_value(nutrition.get("sugar_g"), factor),
            sodium_g=self._scale_value(nutrition.get("sodium_g"), factor),
        )

    def _scale_totals(self, totals: dict[str, Any], multiplier: float) -> NutritionEstimate:
        return NutritionEstimate(
            calories_kcal=self._scale_value(totals.get("calories_kcal"), multiplier),
            mass_g=self._scale_value(totals.get("mass_g"), multiplier),
            protein_g=self._scale_value(totals.get("protein_g"), multiplier),
            carbs_g=self._scale_value(totals.get("carbs_g"), multiplier),
            fat_g=self._scale_value(totals.get("fat_g"), multiplier),
            fiber_g=self._scale_value(totals.get("fiber_g"), multiplier),
            sugar_g=self._scale_value(totals.get("sugar_g"), multiplier),
            sodium_g=self._scale_value(totals.get("sodium_g"), multiplier),
        )

    def _scale_value(self, value: Any, factor: float) -> float | None:
        if value is None:
            return None
        return float(value) * factor

    def _has_any_macro(self, nutrition: dict[str, Any]) -> bool:
        return any(
            nutrition.get(key) is not None
            for key in ("calories_kcal", "protein_g", "carbs_g", "fat_g")
        )

    def _detect_conflicts(
        self,
        vision_data: dict[str, Any],
        text_data: dict[str, Any],
        barcode_data: dict[str, Any],
        conflicts: list[str],
    ) -> None:
        vision_foods = {
            str(item.get("name", "")).lower()
            for item in vision_data.get("detected_items", [])
            if item.get("name")
        }
        text_foods = {
            str(food).lower()
            for food in (text_data.get("parsed") or {}).get("foods", [])
        }
        if vision_foods and text_foods and vision_foods.isdisjoint(text_foods):
            conflicts.append("text_foods_do_not_overlap_vision_detected_items")

        product = barcode_data.get("product") or {}
        if product.get("name") and vision_foods:
            conflicts.append("barcode_product_and_vision_food_may_refer_to_different_sources")

    def _build_items(
        self,
        primary_source: str,
        vision_data: dict[str, Any],
        text_nutrition_data: dict[str, Any],
        barcode_data: dict[str, Any],
        final_macros: NutritionEstimate,
        mass_g: float | None,
        portion_multiplier: float,
        name_overrides: dict[str, str],
        confidence: float,
    ) -> list[MealItemResult]:
        if primary_source == "barcode":
            product = barcode_data.get("product") or {}
            original_name = product.get("name") or "packaged food"
            name = self._corrected_name(original_name, name_overrides)
            notes = ["Nutrition computed from barcode nutrition_per_100g."]
            if name != original_name:
                notes.append(f"Displayed name corrected from {original_name}.")
            return [
                MealItemResult(
                    name=name,
                    source="barcode",
                    mass_g=mass_g,
                    nutrition=final_macros,
                    confidence=confidence,
                    notes=notes,
                )
            ]

        if primary_source == "text_nutrition":
            return self._items_from_text_nutrition(
                text_nutrition_data=text_nutrition_data,
                portion_multiplier=portion_multiplier,
                confidence=confidence,
            )

        if primary_source == "vision" and vision_data.get("ingredients"):
            items = []
            for ingredient in vision_data["ingredients"]:
                nutrition = ingredient.get("nutrition") or {}
                scaled_nutrition = self._scale_totals(nutrition, portion_multiplier)
                original_name = ingredient.get("name", "unknown food")
                name = self._corrected_name(original_name, name_overrides)
                notes = ["Ingredient estimate from vision late fusion used in final macro calculation."]
                if name != original_name:
                    notes.append(f"Displayed name corrected from {original_name}.")
                items.append(
                    MealItemResult(
                        name=name,
                        source="vision",
                        mass_g=scaled_nutrition.mass_g,
                        nutrition=scaled_nutrition,
                        confidence=confidence,
                        notes=notes,
                    )
                )
            return items

        return []

    def _build_barcode_guided_items(
        self,
        barcode_data: dict[str, Any],
        resolution_items: list[dict[str, Any]],
        barcode_nutrition: dict[str, Any],
        fallback_mass_g: float | None,
        portion_multiplier: float,
        name_overrides: dict[str, str],
        confidence: float,
    ) -> tuple[list[MealItemResult], bool]:
        target_index = self._barcode_resolution_target_index(
            barcode_data,
            resolution_items,
        )

        if target_index is None:
            mass_g = fallback_mass_g if fallback_mass_g is not None else 100.0
            nutrition = self._scale_per_100g(
                barcode_nutrition,
                mass_g * portion_multiplier,
            )
            product = barcode_data.get("product") or {}
            original_name = product.get("name") or "packaged food"
            name = self._corrected_name(original_name, name_overrides)
            return [
                MealItemResult(
                    name=name,
                    source="barcode",
                    mass_g=nutrition.mass_g,
                    nutrition=nutrition,
                    confidence=confidence,
                    notes=[
                        "Nutrition computed from barcode nutrition_per_100g.",
                        "No explicit matching text ingredient was found, so fallback meal mass was used.",
                    ],
                )
            ], False

        items: list[MealItemResult] = []
        used_explicit_barcode_mass = False
        for index, item in enumerate(resolution_items):
            if index == target_index:
                target_mass = item.get("mass_g")
                try:
                    mass_g = float(target_mass)
                except (TypeError, ValueError):
                    mass_g = fallback_mass_g if fallback_mass_g is not None else 100.0
                nutrition = self._scale_per_100g(
                    barcode_nutrition,
                    mass_g * portion_multiplier,
                )
                product = barcode_data.get("product") or {}
                name = str(
                    item.get("display_name")
                    or product.get("name")
                    or item.get("canonical_name")
                    or "packaged food"
                )
                if item.get("mass_source") == "explicit_text":
                    used_explicit_barcode_mass = True
                items.append(
                    MealItemResult(
                        name=name,
                        source="barcode",
                        mass_g=nutrition.mass_g,
                        nutrition=nutrition,
                        confidence=confidence,
                        notes=[
                            "Nutrition density came from barcode nutrition_per_100g.",
                            f"Serving mass came from {item.get('mass_source') or 'unknown'} ingredient resolution.",
                        ],
                    )
                )
                continue

            items.append(
                self._item_from_resolution_item(
                    item,
                    confidence=confidence,
                    portion_multiplier=portion_multiplier,
                )
            )

        return items, used_explicit_barcode_mass

    def _build_multi_barcode_guided_items(
        self,
        barcode_data: dict[str, Any],
        resolution_items: list[dict[str, Any]],
        barcode_target_map: dict[int, dict[str, Any]],
        fallback_mass_g: float | None,
        portion_multiplier: float,
        confidence: float,
    ) -> tuple[list[MealItemResult], bool]:
        items: list[MealItemResult] = []
        used_explicit_barcode_mass = False
        for index, item in enumerate(resolution_items):
            barcode_record = barcode_target_map.get(index)
            if barcode_record is None:
                items.append(
                    self._item_from_resolution_item(
                        item,
                        confidence=confidence,
                        portion_multiplier=portion_multiplier,
                    )
                )
                continue

            nutrition_per_100g = barcode_record.get("nutrition_per_100g") or {}
            try:
                mass_g = float(item.get("mass_g"))
            except (TypeError, ValueError):
                mass_g = fallback_mass_g if fallback_mass_g is not None else 100.0
            nutrition = self._scale_per_100g(
                nutrition_per_100g,
                mass_g * portion_multiplier,
            )
            if item.get("mass_source") == "explicit_text":
                used_explicit_barcode_mass = True
            items.append(
                MealItemResult(
                    name=str(
                        item.get("display_name")
                        or (barcode_record.get("product") or {}).get("name")
                        or item.get("canonical_name")
                        or "packaged food"
                    ),
                    source="barcode",
                    mass_g=nutrition.mass_g,
                    nutrition=nutrition,
                    confidence=confidence,
                    notes=[
                        "Nutrition density came from barcode nutrition_per_100g.",
                        f"Barcode product: {(barcode_record.get('product') or {}).get('name') or '-'}",
                        f"Serving mass came from {item.get('mass_source') or 'unknown'} ingredient resolution.",
                    ],
                )
            )
        return items, used_explicit_barcode_mass

    def _barcode_records(self, barcode_data: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = barcode_data.get("candidates") or []
        records: list[dict[str, Any]] = []
        for candidate in candidates:
            if candidate.get("error"):
                continue
            data = candidate.get("data") or {}
            if self._has_any_macro(data.get("nutrition_per_100g") or {}):
                records.append(data)
        if records:
            return records
        if self._has_any_macro(barcode_data.get("nutrition_per_100g") or {}):
            return [barcode_data]
        return []

    def _barcode_target_map(
        self,
        barcode_data: dict[str, Any],
        resolution_items: list[dict[str, Any]],
    ) -> dict[int, dict[str, Any]]:
        target_map: dict[int, dict[str, Any]] = {}
        for record in self._barcode_records(barcode_data):
            target_index = self._barcode_resolution_target_index(record, resolution_items)
            if target_index is None or target_index in target_map:
                continue
            target_map[target_index] = record
        return target_map

    def _resolution_items_from_text_nutrition(
        self,
        text_nutrition_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for item in text_nutrition_data.get("items") or []:
            items.append(
                {
                    "display_name": item.get("name"),
                    "canonical_name": item.get("lookup_name"),
                    "source": "text_nutrition",
                    "mass_g": item.get("mass_g"),
                    "mass_source": item.get("mass_source"),
                    "nutrition": item.get("nutrition") or {},
                    "confidence": item.get("confidence"),
                    "notes": [
                        f"Text nutrition estimate using {item.get('mass_source') or 'unknown'} mass.",
                        *[str(value) for value in item.get("assumptions", [])],
                    ],
                }
            )
        return items

    def _barcode_resolution_target_index(
        self,
        barcode_data: dict[str, Any],
        resolution_items: list[dict[str, Any]],
    ) -> int | None:
        if not resolution_items:
            return None

        product = barcode_data.get("product") or {}
        product_text = " ".join(
            str(value)
            for value in (
                product.get("name"),
                product.get("brand"),
                product.get("categories"),
            )
            if value
        )
        product_tokens = set(self._normalize_food_name(product_text).split())
        if not product_tokens:
            return 0 if len(resolution_items) == 1 else None

        best_index: int | None = None
        best_score = 0
        for index, item in enumerate(resolution_items):
            item_text = " ".join(
                str(value)
                for value in (
                    item.get("display_name"),
                    item.get("canonical_name"),
                )
                if value
            )
            item_tokens = set(self._normalize_food_name(item_text).split())
            score = len(product_tokens & item_tokens)
            if score > best_score:
                best_score = score
                best_index = index

        if best_index is not None and best_score > 0:
            return best_index
        return 0 if len(resolution_items) == 1 else None

    def _item_from_resolution_item(
        self,
        item: dict[str, Any],
        confidence: float,
        portion_multiplier: float,
    ) -> MealItemResult:
        nutrition = self._scale_totals(item.get("nutrition") or {}, portion_multiplier)
        notes = list(item.get("notes") or [])
        links = item.get("links") or []
        for link in links:
            action = link.get("action")
            source = link.get("source")
            name = link.get("name")
            if action and source and name:
                notes.append(f"{action}: {source} '{name}'.")
        return MealItemResult(
            name=str(item.get("display_name") or item.get("canonical_name") or "unknown food"),
            source=(
                "text_nutrition"
                if item.get("source") in {"text_nutrition", "mixed"}
                else str(item.get("source") or "vision")
            ),
            mass_g=nutrition.mass_g,
            nutrition=nutrition,
            confidence=min(float(item.get("confidence") or confidence), confidence),
            notes=notes,
        )

    def _build_mixed_text_vision_items(
        self,
        vision_data: dict[str, Any],
        text_nutrition_data: dict[str, Any],
        portion_multiplier: float,
        name_overrides: dict[str, str],
        anchor: NutritionEstimate,
        confidence: float,
        warnings: list[str],
    ) -> list[MealItemResult]:
        explicit_text_items = self._items_from_text_nutrition(
            text_nutrition_data=text_nutrition_data,
            portion_multiplier=portion_multiplier,
            confidence=confidence,
            mass_sources={"explicit_text"},
        )
        explicit_tokens = self._text_nutrition_tokens(
            text_nutrition_data,
            mass_sources={"explicit_text"},
        )
        replacement_map = self._vision_text_replacement_map(
            vision_data=vision_data,
            text_nutrition_data=text_nutrition_data,
            name_overrides=name_overrides,
        )
        explicit_total = self._sum_item_nutrition(explicit_text_items)

        remaining_vision_items = self._remaining_vision_items(
            vision_data=vision_data,
            portion_multiplier=portion_multiplier,
            name_overrides=name_overrides,
            covered_tokens=explicit_tokens,
            replacement_map=replacement_map,
            confidence=confidence,
        )
        if not remaining_vision_items:
            return explicit_text_items

        remaining_anchor = self._remaining_anchor(anchor, explicit_total, warnings)
        scaled_vision_items = self._scale_items_to_anchor(
            remaining_vision_items,
            remaining_anchor,
            note="Scaled to the remaining vision mass/calorie budget after explicit text quantities.",
        )
        return [*explicit_text_items, *scaled_vision_items]

    def _should_use_text_guided_vision_items(
        self,
        primary_source: str,
        vision_data: dict[str, Any],
        text_nutrition_data: dict[str, Any],
        text_foods: list[str],
        hidden_ingredients: list[Any],
        name_overrides: dict[str, str],
        llm_decision: dict[str, Any] | None,
    ) -> bool:
        if primary_source != "vision" or not text_foods:
            return False
        if not self._has_any_macro(vision_data.get("totals") or {}):
            return False
        if name_overrides:
            return False
        if self._llm_rule_contains(llm_decision, "preserve vision"):
            return False
        if self._llm_rule_contains(llm_decision, "kept vision"):
            return False
        if self._has_explicit_text_mass(text_nutrition_data):
            return True

        vision_names = [
            str(ingredient.get("name"))
            for ingredient in vision_data.get("ingredients", [])
            if ingredient.get("name")
        ]
        if not vision_names:
            return False

        vision_tokens = {self._normalize_food_name(name) for name in vision_names}
        text_tokens = {self._normalize_food_name(food) for food in text_foods}
        hidden_tokens = {
            self._normalize_food_name(str(ingredient))
            for ingredient in hidden_ingredients
        }

        missing_text_foods = text_tokens - vision_tokens
        extra_vision_foods = vision_tokens - text_tokens
        return bool(missing_text_foods or extra_vision_foods or (hidden_tokens & text_tokens))

    def _build_text_guided_vision_items(
        self,
        text_foods: list[str],
        final_macros: NutritionEstimate,
        text_nutrition_data: dict[str, Any],
        confidence: float,
    ) -> list[MealItemResult]:
        text_items = self._items_from_text_nutrition(
            text_nutrition_data=text_nutrition_data,
            portion_multiplier=1.0,
            confidence=confidence,
        )
        if text_items:
            return self._scale_items_to_anchor(text_items, final_macros)

        if not text_foods:
            return []

        total_mass = final_macros.mass_g or 0.0
        total_calories = final_macros.calories_kcal
        mass_per_food = total_mass / len(text_foods) if total_mass > 0 else None

        raw_items: list[tuple[str, NutritionEstimate, bool]] = []
        for food in text_foods:
            prior = self.nutrition_lookup.lookup(food)
            mass_g = mass_per_food
            raw_calories = (
                mass_g * prior.calories_per_g
                if mass_g is not None
                else None
            )
            raw_items.append(
                (
                    food,
                    NutritionEstimate(
                        calories_kcal=raw_calories,
                        mass_g=mass_g,
                        protein_g=(
                            mass_g * prior.protein_per_g
                            if mass_g is not None
                            else None
                        ),
                        carbs_g=(
                            mass_g * prior.carbs_per_g
                            if mass_g is not None
                            else None
                        ),
                        fat_g=(
                            mass_g * prior.fat_per_g
                            if mass_g is not None
                            else None
                        ),
                    ),
                    prior.matched,
                )
            )

        raw_calories_total = sum(
            item.calories_kcal or 0.0
            for _, item, _ in raw_items
        )
        calorie_scale = (
            total_calories / raw_calories_total
            if total_calories is not None and raw_calories_total > 0
            else 1.0
        )

        items = []
        for food, nutrition, matched in raw_items:
            scaled_nutrition = NutritionEstimate(
                calories_kcal=self._scale_value(nutrition.calories_kcal, calorie_scale),
                mass_g=nutrition.mass_g,
                protein_g=self._scale_value(nutrition.protein_g, calorie_scale),
                carbs_g=self._scale_value(nutrition.carbs_g, calorie_scale),
                fat_g=self._scale_value(nutrition.fat_g, calorie_scale),
            )
            notes = [
                "Text-confirmed ingredient allocated from vision total mass and calorie anchor."
            ]
            if not matched:
                notes.append("Used generic nutrition prior because no close food match was found.")
            items.append(
                MealItemResult(
                    name=food,
                    source="text",
                    mass_g=scaled_nutrition.mass_g,
                    nutrition=scaled_nutrition,
                    confidence=confidence,
                    notes=notes,
                )
            )
        return items

    def _items_from_text_nutrition(
        self,
        text_nutrition_data: dict[str, Any],
        portion_multiplier: float,
        confidence: float,
        mass_sources: set[str] | None = None,
    ) -> list[MealItemResult]:
        items = []
        for item in text_nutrition_data.get("items") or []:
            if mass_sources is not None and item.get("mass_source") not in mass_sources:
                continue
            nutrition = self._scale_totals(item.get("nutrition") or {}, portion_multiplier)
            items.append(
                MealItemResult(
                    name=str(item.get("name") or item.get("lookup_name") or "unknown food"),
                    source="text_nutrition",
                    mass_g=nutrition.mass_g,
                    nutrition=nutrition,
                    confidence=min(float(item.get("confidence") or confidence), confidence),
                    notes=[
                        f"Text nutrition estimate using {item.get('mass_source') or 'unknown'} mass.",
                        *[str(value) for value in item.get("assumptions", [])],
                    ],
                )
            )
        return items

    def _items_from_resolution(
        self,
        ingredient_resolution_data: dict[str, Any],
        confidence: float,
    ) -> list[MealItemResult]:
        items = []
        for item in ingredient_resolution_data.get("items") or []:
            nutrition = NutritionEstimate.model_validate(item.get("nutrition") or {})
            notes = list(item.get("notes") or [])
            links = item.get("links") or []
            for link in links:
                action = link.get("action")
                source = link.get("source")
                name = link.get("name")
                if action and source and name:
                    notes.append(f"{action}: {source} '{name}'.")
            items.append(
                MealItemResult(
                    name=str(item.get("display_name") or item.get("canonical_name") or "unknown food"),
                    source=(
                        "text_nutrition"
                        if item.get("source") in {"text_nutrition", "mixed"}
                        else str(item.get("source") or "vision")
                    ),
                    mass_g=nutrition.mass_g,
                    nutrition=nutrition,
                    confidence=min(float(item.get("confidence") or confidence), confidence),
                    notes=notes,
                )
            )
        return items

    def _resolution_uses_only_explicit_text(self, ingredient_resolution_data: dict[str, Any]) -> bool:
        items = ingredient_resolution_data.get("items") or []
        if not items:
            return False
        return all(
            item.get("mass_source") == "explicit_text"
            and item.get("source") in {"text_nutrition", "mixed"}
            for item in items
        )

    def _scale_items_to_anchor(
        self,
        items: list[MealItemResult],
        anchor: NutritionEstimate,
        note: str = "Scaled to the selected vision calorie/mass anchor.",
    ) -> list[MealItemResult]:
        item_total = self._sum_item_nutrition(items)
        calorie_scale = (
            anchor.calories_kcal / item_total.calories_kcal
            if anchor.calories_kcal is not None and item_total.calories_kcal
            else 1.0
        )
        mass_scale = (
            anchor.mass_g / item_total.mass_g
            if anchor.mass_g is not None and item_total.mass_g
            else 1.0
        )
        scaled = []
        for item in items:
            scaled_nutrition = NutritionEstimate(
                calories_kcal=self._scale_value(item.nutrition.calories_kcal, calorie_scale),
                mass_g=self._scale_value(item.nutrition.mass_g, mass_scale),
                protein_g=self._scale_value(item.nutrition.protein_g, calorie_scale),
                carbs_g=self._scale_value(item.nutrition.carbs_g, calorie_scale),
                fat_g=self._scale_value(item.nutrition.fat_g, calorie_scale),
            )
            scaled.append(
                MealItemResult(
                    name=item.name,
                    source=item.source,
                    mass_g=scaled_nutrition.mass_g,
                    nutrition=scaled_nutrition,
                    confidence=item.confidence,
                    notes=[
                        *item.notes,
                        note,
                    ],
                )
            )
        return scaled

    def _remaining_vision_items(
        self,
        vision_data: dict[str, Any],
        portion_multiplier: float,
        name_overrides: dict[str, str],
        covered_tokens: set[str],
        replacement_map: dict[str, str],
        confidence: float,
    ) -> list[MealItemResult]:
        items = []
        for ingredient in vision_data.get("ingredients") or []:
            original_name = str(ingredient.get("name") or "unknown food")
            corrected_name = self._corrected_name(original_name, name_overrides)
            if self._normalize_food_name(original_name) in replacement_map:
                continue
            if self._normalize_food_name(original_name) in covered_tokens:
                continue
            if self._normalize_food_name(corrected_name) in covered_tokens:
                continue

            nutrition = ingredient.get("nutrition") or {}
            scaled_nutrition = self._scale_totals(nutrition, portion_multiplier)
            notes = ["Remaining visible ingredient from vision after text quantities were applied."]
            if corrected_name != original_name:
                notes.append(f"Displayed name corrected from {original_name}.")
            items.append(
                MealItemResult(
                    name=corrected_name,
                    source="vision",
                    mass_g=scaled_nutrition.mass_g,
                    nutrition=scaled_nutrition,
                    confidence=confidence,
                    notes=notes,
                )
            )
        return items

    def _vision_text_replacement_map(
        self,
        vision_data: dict[str, Any],
        text_nutrition_data: dict[str, Any],
        name_overrides: dict[str, str],
    ) -> dict[str, str]:
        explicit_items = [
            item
            for item in text_nutrition_data.get("items") or []
            if item.get("mass_source") == "explicit_text"
        ]
        vision_names = [
            str(ingredient.get("name"))
            for ingredient in vision_data.get("ingredients") or []
            if ingredient.get("name")
        ]
        if not explicit_items or not vision_names:
            return {}

        explicit_tokens = self._text_nutrition_tokens(
            text_nutrition_data,
            mass_sources={"explicit_text"},
        )
        replacement_map: dict[str, str] = {}

        for vision_name in vision_names:
            corrected_name = self._corrected_name(vision_name, name_overrides)
            corrected_token = self._normalize_food_name(corrected_name)
            original_token = self._normalize_food_name(vision_name)
            if corrected_token in explicit_tokens:
                replacement_map[original_token] = corrected_token

        if replacement_map:
            return replacement_map

        if len(vision_names) == 1 and len(explicit_items) == 1:
            vision_token = self._normalize_food_name(vision_names[0])
            text_token = self._normalize_food_name(
                str(explicit_items[0].get("lookup_name") or explicit_items[0].get("name"))
            )
            return {vision_token: text_token}

        return {}

    def _remaining_anchor(
        self,
        anchor: NutritionEstimate,
        used: NutritionEstimate,
        warnings: list[str],
    ) -> NutritionEstimate:
        if (
            anchor.mass_g is not None
            and used.mass_g is not None
            and used.mass_g > anchor.mass_g
        ):
            warnings.append("Explicit text mass exceeds vision total mass; final mass may exceed vision estimate.")
        if (
            anchor.calories_kcal is not None
            and used.calories_kcal is not None
            and used.calories_kcal > anchor.calories_kcal
        ):
            warnings.append("Explicit text calories exceed vision total calories; final calories may exceed vision estimate.")

        return NutritionEstimate(
            calories_kcal=self._subtract_floor(anchor.calories_kcal, used.calories_kcal),
            mass_g=self._subtract_floor(anchor.mass_g, used.mass_g),
            protein_g=self._subtract_floor(anchor.protein_g, used.protein_g),
            carbs_g=self._subtract_floor(anchor.carbs_g, used.carbs_g),
            fat_g=self._subtract_floor(anchor.fat_g, used.fat_g),
            fiber_g=self._subtract_floor(anchor.fiber_g, used.fiber_g),
            sugar_g=self._subtract_floor(anchor.sugar_g, used.sugar_g),
            sodium_g=self._subtract_floor(anchor.sodium_g, used.sodium_g),
        )

    def _subtract_floor(self, left: float | None, right: float | None) -> float | None:
        if left is None:
            return None
        return max(float(left) - float(right or 0.0), 0.0)

    def _sum_item_nutrition(self, items: list[MealItemResult]) -> NutritionEstimate:
        return NutritionEstimate(
            calories_kcal=self._sum_optional(item.nutrition.calories_kcal for item in items),
            mass_g=self._sum_optional(item.nutrition.mass_g for item in items),
            protein_g=self._sum_optional(item.nutrition.protein_g for item in items),
            carbs_g=self._sum_optional(item.nutrition.carbs_g for item in items),
            fat_g=self._sum_optional(item.nutrition.fat_g for item in items),
            fiber_g=self._sum_optional(item.nutrition.fiber_g for item in items),
            sugar_g=self._sum_optional(item.nutrition.sugar_g for item in items),
            sodium_g=self._sum_optional(item.nutrition.sodium_g for item in items),
        )

    def _sum_optional(self, values: Any) -> float | None:
        present = [float(value) for value in values if value is not None]
        return sum(present) if present else None

    def _has_explicit_text_mass(self, text_nutrition_data: dict[str, Any]) -> bool:
        items = text_nutrition_data.get("items") or []
        return bool(items) and all(item.get("mass_source") == "explicit_text" for item in items)

    def _has_any_explicit_text_mass(self, text_nutrition_data: dict[str, Any]) -> bool:
        return any(
            item.get("mass_source") == "explicit_text"
            for item in text_nutrition_data.get("items") or []
        )

    def _explicit_text_covers_vision(
        self,
        text_nutrition_data: dict[str, Any],
        vision_data: dict[str, Any],
        llm_decision: dict[str, Any] | None,
    ) -> bool:
        covered_tokens = self._text_nutrition_tokens(
            text_nutrition_data,
            mass_sources={"explicit_text"},
        )
        if not covered_tokens:
            return False
        explicit_items = [
            item
            for item in text_nutrition_data.get("items") or []
            if item.get("mass_source") == "explicit_text"
        ]
        vision_items = [
            ingredient
            for ingredient in vision_data.get("ingredients") or []
            if ingredient.get("name")
        ]
        if len(explicit_items) == 1 and len(vision_items) == 1:
            return True

        name_overrides = (llm_decision or {}).get("corrected_food_names") or {}
        for ingredient in vision_items:
            original_name = str(ingredient.get("name") or "")
            corrected_name = self._corrected_name(original_name, name_overrides)
            if self._normalize_food_name(original_name) not in covered_tokens and (
                self._normalize_food_name(corrected_name) not in covered_tokens
            ):
                return False
        return True

    def _text_nutrition_tokens(
        self,
        text_nutrition_data: dict[str, Any],
        mass_sources: set[str] | None = None,
    ) -> set[str]:
        tokens: set[str] = set()
        for item in text_nutrition_data.get("items") or []:
            if mass_sources is not None and item.get("mass_source") not in mass_sources:
                continue
            for key in ("name", "lookup_name"):
                value = item.get(key)
                if value:
                    tokens.add(self._normalize_food_name(str(value)))
        return tokens

    def _name_overrides(
        self,
        vision_data: dict[str, Any],
        text_data: dict[str, Any],
        barcode_data: dict[str, Any],
        llm_decision: dict[str, Any] | None,
        ingredient_resolution_data: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        accepted_replacements = {
            str(key): str(value)
            for key, value in ((ingredient_resolution_data or {}).get("replacements") or {}).items()
            if key and value
        }
        if ingredient_resolution_data is not None:
            return accepted_replacements

        if ingredient_resolution_data is None:
            overrides = {
                str(key): str(value)
                for key, value in ((llm_decision or {}).get("corrected_food_names") or {}).items()
                if key and value
            }
            if overrides:
                return overrides

        vision_names = [
            str(ingredient.get("name"))
            for ingredient in vision_data.get("ingredients", [])
            if ingredient.get("name")
        ]
        text_foods = self._text_foods(text_data)
        if not vision_names or not text_foods:
            return {}

        vision_tokens = {self._normalize_food_name(name) for name in vision_names}
        text_tokens = {self._normalize_food_name(name) for name in text_foods}
        if vision_tokens & text_tokens:
            return {}

        if len(text_foods) == 1 and len(vision_names) == 1 and not ingredient_resolution_data:
            return {vision_names[0]: text_foods[0]}

        product = barcode_data.get("product") or {}
        product_name = product.get("name")
        if product_name and len(text_foods) == 1:
            return {str(product_name): text_foods[0]}

        return {}

    def _corrected_name(self, name: str, overrides: dict[str, str]) -> str:
        if name in overrides:
            return overrides[name]
        normalized = self._normalize_food_name(name)
        for original, corrected in overrides.items():
            if self._normalize_food_name(original) == normalized:
                return corrected
        return name

    def _text_foods(self, text_data: dict[str, Any]) -> list[str]:
        parsed = text_data.get("parsed") or {}
        candidates = parsed.get("foods") or []
        if not candidates:
            raw_text = (text_data.get("input") or {}).get("raw_text") or ""
            candidates = [raw_text]

        entity_confidence = self._food_entity_confidence(text_data)
        foods: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            for part in re.split(r"\s*(?:,|&|\band\b|\bve\b|\+)\s*", str(candidate), flags=re.IGNORECASE):
                cleaned = part.strip(" .;:")
                normalized = self._normalize_food_name(cleaned)
                confidence = entity_confidence.get(normalized)
                if confidence is not None and confidence < 0.5:
                    continue
                if len(normalized) < 3:
                    continue
                if cleaned and normalized and normalized not in seen:
                    foods.append(cleaned)
                    seen.add(normalized)
        return foods

    def _normalize_food_name(self, name: str) -> str:
        return re.sub(r"[^a-z0-9çğıöşü]+", " ", name.lower()).strip()

    def _food_entity_confidence(self, text_data: dict[str, Any]) -> dict[str, float]:
        confidences: dict[str, float] = {}
        for entity in text_data.get("entities") or []:
            if entity.get("type") != "food":
                continue
            normalized = self._normalize_food_name(str(entity.get("text", "")))
            if not normalized:
                continue
            confidence = entity.get("confidence")
            if confidence is None:
                continue
            confidences[normalized] = max(confidences.get(normalized, 0.0), float(confidence))
        return confidences

    def _llm_rule_contains(
        self,
        llm_decision: dict[str, Any] | None,
        needle: str,
    ) -> bool:
        if not llm_decision:
            return False
        normalized_needle = needle.lower()
        values = [
            *llm_decision.get("applied_rules", []),
            *llm_decision.get("assumptions", []),
            *llm_decision.get("conflicts", []),
        ]
        if llm_decision.get("explanation"):
            values.append(llm_decision["explanation"])
        return any(normalized_needle in str(value).lower() for value in values)

    def _meal_name(
        self,
        text_data: dict[str, Any],
        vision_data: dict[str, Any],
        barcode_data: dict[str, Any],
    ) -> str | None:
        product = barcode_data.get("product") or {}
        if product.get("name"):
            return product["name"]

        foods = (text_data.get("parsed") or {}).get("foods") or []
        if foods:
            return ", ".join(foods[:3])

        detected = vision_data.get("detected_items") or []
        if detected:
            return ", ".join(str(item.get("name")) for item in detected[:3])

        return None

    def _confidence(
        self,
        outputs: dict[str, AgentResponse | None],
        conflicts: list[str] | None = None,
        warnings: list[str] | None = None,
        text_guided_items: bool = False,
    ) -> float:
        confidences = [
            output.confidence
            for output in outputs.values()
            if output is not None and output.error is None
        ]
        if not confidences:
            return 0.0
        confidence = sum(confidences) / len(confidences)
        confidence -= min(len(conflicts or []) * 0.06, 0.18)
        confidence -= min(len(warnings or []) * 0.04, 0.12)
        if text_guided_items:
            confidence -= 0.05
        return round(min(max(confidence, 0.0), 0.95), 4)

    def _reasoning_summary(
        self,
        decision: FusionDecision,
        warnings: list[str],
    ) -> str:
        parts = [
            f"Primary nutrition source: {decision.primary_nutrition_source}.",
            f"Mass source: {decision.mass_source}.",
            f"Portion multiplier: {decision.portion_multiplier}.",
        ]
        if decision.applied_rules:
            parts.append("Applied rules: " + ", ".join(decision.applied_rules) + ".")
        if decision.assumptions:
            parts.append("Assumptions: " + " ".join(decision.assumptions))
        if decision.conflicts:
            parts.append("Conflicts: " + ", ".join(decision.conflicts) + ".")
        if warnings:
            parts.append("Warnings: " + " ".join(warnings))
        return " ".join(parts)
