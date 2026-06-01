"""Orchestrator agent for collecting modality-agent evidence."""

from __future__ import annotations

import asyncio
from typing import Any

from agents.base_agent import BaseAgent
from agents.orchestrator.tools import analyze_barcode, analyze_text, analyze_vision
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

        if input_summary.image_provided:
            vision_payload = {
                "image_path": payload.get("image_path"),
                "image_bytes": payload.get("image_bytes") or payload.get("bytes"),
                "filename": payload.get("image_filename") or payload.get("filename"),
                "content_type": payload.get("image_content_type"),
                "depth_path": payload.get("depth_path"),
                "depth_bytes": payload.get("depth_bytes"),
            }
            tasks["vision"] = asyncio.create_task(analyze_vision(vision_payload))

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
            "vision": None,
            "text": None,
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

        contract = OrchestratorAgentData(
            input=input_summary,
            tool_order=list(tasks.keys()),
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
        if key == "text":
            return {
                "raw_text": (data.get("input") or {}).get("raw_text"),
                "parsed": data.get("parsed", {}),
                "entities": data.get("entities", []),
            }
        if key == "barcode":
            return {
                "barcode": data.get("barcode"),
                "product": data.get("product"),
                "nutrition_per_100g": data.get("nutrition_per_100g"),
            }
        return data
