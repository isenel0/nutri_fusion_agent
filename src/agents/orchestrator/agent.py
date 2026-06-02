"""Orchestrator agent for collecting modality-agent evidence."""

from __future__ import annotations

import asyncio
from typing import Any

from agents.base_agent import BaseAgent
from agents.orchestrator.tools import (
    analyze_barcode,
    analyze_text,
    analyze_text_nutrition,
    analyze_vision,
    generate_depth,
    resolve_ingredients,
)
from schemas import (
    AgentResponse,
    OrchestratorAgentData,
    OrchestratorInputSummary,
)


class OrchestratorAgent(BaseAgent):
    """Runs available modality tools and returns a structured evidence bundle."""

    async def process(self, input_data: Any) -> AgentResponse:
        payload = input_data if isinstance(input_data, dict) else {}
        input_summary = self._input_summary(payload)

        tasks: dict[str, asyncio.Task[AgentResponse]] = {}

        if input_summary.text_provided:
            tasks["text"] = asyncio.create_task(
                analyze_text({"text": payload.get("text")})
            )

        if input_summary.barcode_text_provided or input_summary.barcode_image_provided:
            barcode_payload = {
                "barcode": payload.get("barcode"),
                "image_path": payload.get("barcode_image_path"),
                "image_bytes": payload.get("barcode_image_bytes"),
                "image_filename": payload.get("barcode_image_filename"),
            }
            tasks["barcode"] = asyncio.create_task(analyze_barcode(barcode_payload))

        outputs: dict[str, AgentResponse | None] = {
            "depth": None,
            "vision": None,
            "text": None,
            "text_nutrition": None,
            "ingredient_resolution": None,
            "barcode": None,
        }
        errors: dict[str, str] = {}

        if tasks:
            gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for key, result in zip(tasks.keys(), gathered):
                if isinstance(result, Exception):
                    message = f"{key} tool failed: {result}"
                    errors[key] = message
                    outputs[key] = AgentResponse(
                        source=key,
                        confidence=0.0,
                        data={},
                        error=message,
                    )
                else:
                    outputs[key] = result
                    if result.error:
                        errors[key] = result.error

        task_order = list(tasks.keys())

        if input_summary.image_provided:
            vision_payload = {
                "image_path": payload.get("image_path"),
                "image_bytes": payload.get("image_bytes") or payload.get("bytes"),
                "filename": payload.get("image_filename") or payload.get("filename"),
                "content_type": payload.get("image_content_type"),
                "depth_path": payload.get("depth_path"),
                "depth_bytes": payload.get("depth_bytes"),
            }

            if not input_summary.depth_image_provided:
                depth_payload = {
                    "image_path": payload.get("image_path"),
                    "image_bytes": payload.get("image_bytes") or payload.get("bytes"),
                    "image_filename": payload.get("image_filename") or payload.get("filename"),
                }
                depth_result = await generate_depth(depth_payload)
                outputs["depth"] = depth_result
                task_order.append("depth")
                if depth_result.error:
                    errors["depth"] = depth_result.error
                elif depth_result.data.get("depth_path"):
                    vision_payload["depth_path"] = depth_result.data["depth_path"]

            vision_result = await analyze_vision(vision_payload)
            outputs["vision"] = vision_result
            task_order.append("vision")
            if vision_result.error:
                errors["vision"] = vision_result.error

        if outputs.get("text") is not None:
            try:
                text_nutrition = await analyze_text_nutrition(
                    {
                        "text": outputs.get("text"),
                        "vision": outputs.get("vision"),
                    }
                )
                outputs["text_nutrition"] = text_nutrition
                if text_nutrition.error:
                    errors["text_nutrition"] = text_nutrition.error
                task_order = [*task_order, "text_nutrition"]
            except Exception as exc:
                message = f"text_nutrition tool failed: {exc}"
                errors["text_nutrition"] = message
                outputs["text_nutrition"] = AgentResponse(
                    source="text_nutrition",
                    confidence=0.0,
                    data={},
                    error=message,
                )
                task_order = [*tasks.keys(), "text_nutrition"]

        if outputs.get("vision") is not None or outputs.get("text_nutrition") is not None:
            try:
                ingredient_resolution = await resolve_ingredients(
                    {
                        "vision": outputs.get("vision"),
                        "text_nutrition": outputs.get("text_nutrition"),
                    }
                )
                outputs["ingredient_resolution"] = ingredient_resolution
                if ingredient_resolution.error:
                    errors["ingredient_resolution"] = ingredient_resolution.error
                task_order = [*task_order, "ingredient_resolution"]
            except Exception as exc:
                message = f"ingredient_resolution tool failed: {exc}"
                errors["ingredient_resolution"] = message
                outputs["ingredient_resolution"] = AgentResponse(
                    source="ingredient_resolution",
                    confidence=0.0,
                    data={},
                    error=message,
                )
                task_order = [*task_order, "ingredient_resolution"]

        contract = OrchestratorAgentData(
            input=input_summary,
            tool_order=task_order,
            outputs=outputs,
            errors=errors,
            llm_context=self._llm_context(outputs),
            llm_notes=[
                "This is an evidence bundle only; final nutrition fusion is not applied here.",
                "Use contract_version fields inside each output to route deterministic fusion rules.",
            ],
        )

        return AgentResponse(
            source="orchestrator",
            confidence=self._bundle_confidence(outputs),
            data=contract.model_dump(),
            error=None if tasks else "No modality inputs were provided.",
        )

    def _input_summary(self, payload: dict[str, Any]) -> OrchestratorInputSummary:
        return OrchestratorInputSummary(
            image_provided=bool(
                payload.get("image_path")
                or payload.get("image_bytes")
                or payload.get("bytes")
            ),
            depth_image_provided=bool(
                payload.get("depth_path")
                or payload.get("depth_bytes")
            ),
            text_provided=bool(payload.get("text")),
            barcode_text_provided=bool(payload.get("barcode")),
            barcode_image_provided=bool(
                payload.get("barcode_image_path")
                or payload.get("barcode_image_bytes")
            ),
        )

    def _bundle_confidence(self, outputs: dict[str, AgentResponse | None]) -> float:
        confidences = [
            output.confidence
            for output in outputs.values()
            if output is not None and output.error is None
        ]
        if not confidences:
            return 0.0
        return sum(confidences) / len(confidences)

    def _llm_context(
        self,
        outputs: dict[str, AgentResponse | None],
    ) -> dict[str, Any]:
        context: dict[str, Any] = {}
        for key, output in outputs.items():
            if output is None:
                continue
            data = output.data
            context[key] = {
                "contract_version": data.get("contract_version"),
                "confidence": output.confidence,
                "error": output.error,
                "summary": self._summary_for_output(key, data),
            }
        return context

    def _summary_for_output(self, key: str, data: dict[str, Any]) -> dict[str, Any]:
        if key == "vision":
            return {
                "mode": data.get("mode"),
                "detected_items": data.get("detected_items", []),
                "ingredients": data.get("ingredients", []),
                "totals": data.get("totals"),
                "missing_requirements": data.get("missing_requirements", []),
            }
        if key == "depth":
            return {
                "mode": data.get("mode"),
                "model_name": data.get("model_name"),
                "depth_path": data.get("depth_path"),
                "output_format": data.get("output_format"),
            }
        if key == "text":
            return {
                "raw_text": (data.get("input") or {}).get("raw_text"),
                "parsed": data.get("parsed", {}),
                "entities": data.get("entities", []),
            }
        if key == "text_nutrition":
            return {
                "items": data.get("items", []),
                "totals": data.get("totals"),
                "assumptions": data.get("assumptions", []),
                "warnings": data.get("warnings", []),
            }
        if key == "ingredient_resolution":
            return {
                "items": data.get("items", []),
                "totals": data.get("totals"),
                "replacements": data.get("replacements", {}),
                "unmatched_vision": data.get("unmatched_vision", []),
                "unmatched_text": data.get("unmatched_text", []),
                "assumptions": data.get("assumptions", []),
                "warnings": data.get("warnings", []),
            }
        if key == "barcode":
            return {
                "barcode": data.get("barcode"),
                "product": data.get("product"),
                "nutrition_per_100g": data.get("nutrition_per_100g"),
            }
        return data
