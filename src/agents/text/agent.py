"""Text agent boilerplate for user-described meal parsing."""

from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from schemas import AgentResponse


class TextAgent(BaseAgent):
    """Parses meal details from free-form text input.

    Replace this placeholder with NLP extraction and normalization logic.
    """

    async def process(self, input_data: Any) -> AgentResponse:
        # TODO: Integrate prompt parsing / entity extraction pipeline.
        return AgentResponse(
            source="text",
            confidence=0.7,
            data={
                "food_name": "grilled_chicken_with_rice",
                "portion_size": "250 g",
                "macros": {
                    "calories_kcal": 430,
                    "protein_g": 32,
                    "carbs_g": 45,
                    "fat_g": 12,
                },
                "notes": "Dummy output from TextAgent.",
            },
            error=None,
        )
