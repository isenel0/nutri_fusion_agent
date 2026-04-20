"""Fusion agent boilerplate for multimodal reasoning and conflict resolution."""

from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from schemas import AgentResponse


class FusionAgent(BaseAgent):
    """Synthesizes outputs from vision, text, and barcode agents.

    In production, this can call an LLM to reconcile conflicts, such as
    raw-vs-cooked weight assumptions and duplicated food entities.
    """

    async def process(self, input_data: Any) -> AgentResponse:
        # Expected input_data shape:
        # {
        #   "vision": AgentResponse | None,
        #   "text": AgentResponse | None,
        #   "barcode": AgentResponse | None,
        # }
        outputs = input_data if isinstance(input_data, dict) else {}

        vision = outputs.get("vision")
        text = outputs.get("text")
        barcode = outputs.get("barcode")

        # TODO: Replace with weighted voting + LLM adjudication.
        return AgentResponse(
            source="fusion",
            confidence=0.85,
            data={
                "final_macros": {
                    "calories_kcal": 400,
                    "protein_g": 25,
                    "carbs_g": 38,
                    "fat_g": 14,
                },
                "reasoning_summary": (
                    "Dummy fusion result. Prioritized text macros, then adjusted "
                    "with barcode packaged nutrition and vision portion cues."
                ),
                "inputs_used": {
                    "vision": bool(vision),
                    "text": bool(text),
                    "barcode": bool(barcode),
                },
            },
            error=None,
        )
