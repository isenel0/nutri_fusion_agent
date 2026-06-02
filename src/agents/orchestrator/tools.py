"""Tool wrappers around existing modality agents.

These functions are intentionally framework-neutral. They can be wrapped by
LangChain, OpenAI tool-calling, or a custom orchestrator without changing the
underlying agents.
"""

from __future__ import annotations

from typing import Any

from agents.barcode.agent import BarcodeAgent
from agents.depth.agent import DepthAgent
from agents.food.resolver import IngredientResolverAgent
from agents.text.agent import TextAgent
from agents.text.nutrition_agent import TextNutritionAgent
from agents.vision.agent import VisionAgent
from schemas import AgentResponse

_TEXT_NUTRITION_AGENT = TextNutritionAgent()


async def analyze_vision(payload: dict[str, Any]) -> AgentResponse:
    """Run the vision agent over image/depth input."""
    return await VisionAgent().process(payload)


async def generate_depth(payload: dict[str, Any]) -> AgentResponse:
    """Generate a depth image from RGB input."""
    return await DepthAgent().process(payload)


async def analyze_barcode(payload: str | dict[str, Any]) -> AgentResponse:
    """Run the barcode agent over barcode text or image input."""
    return await BarcodeAgent().process(payload)


async def analyze_text(payload: str | dict[str, Any]) -> AgentResponse:
    """Run the text agent over raw meal context."""
    return await TextAgent().process(payload)


async def analyze_text_nutrition(payload: dict[str, Any]) -> AgentResponse:
    """Resolve text foods and quantities into nutrition estimates."""
    return await _TEXT_NUTRITION_AGENT.process(payload)


async def resolve_ingredients(payload: dict[str, Any]) -> AgentResponse:
    """Resolve ingredient identities across vision and text nutrition evidence."""
    return await IngredientResolverAgent().process(payload)


TOOL_SPECS: dict[str, dict[str, Any]] = {
    "analyze_vision": {
        "description": "Extract visible food ratios and optional late-fusion nutrition from image/depth input.",
        "returns_contract": "vision.v1",
    },
    "generate_depth": {
        "description": "Generate a relative depth image from RGB input using Depth Anything V2.",
        "returns_contract": "depth.v1",
    },
    "analyze_barcode": {
        "description": "Resolve packaged-product nutrition from barcode text or barcode image input.",
        "returns_contract": "barcode.v1",
    },
    "analyze_text": {
        "description": "Extract food, quantity, unit, method, and modifier entities from user meal text.",
        "returns_contract": "text.v1",
    },
    "analyze_text_nutrition": {
        "description": "Estimate calories/macros from text foods and Turkish quantity expressions.",
        "returns_contract": "text_nutrition.v1",
    },
    "resolve_ingredients": {
        "description": "Resolve canonical ingredients and replacements across vision/text evidence.",
        "returns_contract": "ingredient_resolution.v1",
    },
}
