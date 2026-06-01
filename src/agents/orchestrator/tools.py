"""Tool wrappers around existing modality agents.

These functions are intentionally framework-neutral. They can be wrapped by
LangChain, OpenAI tool-calling, or a custom orchestrator without changing the
underlying agents.
"""

from __future__ import annotations

from typing import Any

from agents.barcode.agent import BarcodeAgent
from agents.text.agent import TextAgent
from agents.vision.agent import VisionAgent
from schemas import AgentResponse


async def analyze_vision(payload: dict[str, Any]) -> AgentResponse:
    """Run the vision agent over image/depth input."""
    return await VisionAgent().process(payload)


async def analyze_barcode(payload: str | dict[str, Any]) -> AgentResponse:
    """Run the barcode agent over barcode text or image input."""
    return await BarcodeAgent().process(payload)


async def analyze_text(payload: str | dict[str, Any]) -> AgentResponse:
    """Run the text agent over raw meal context."""
    return await TextAgent().process(payload)


TOOL_SPECS: dict[str, dict[str, Any]] = {
    "analyze_vision": {
        "description": "Extract visible food ratios and optional late-fusion nutrition from image/depth input.",
        "returns_contract": "vision.v1",
    },
    "analyze_barcode": {
        "description": "Resolve packaged-product nutrition from barcode text or barcode image input.",
        "returns_contract": "barcode.v1",
    },
    "analyze_text": {
        "description": "Extract food, quantity, unit, method, and modifier entities from user meal text.",
        "returns_contract": "text.v1",
    },
}
