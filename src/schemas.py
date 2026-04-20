"""Shared Pydantic schemas used across all agents.

Every agent in the pipeline should return an AgentResponse so the
fusion layer can consume outputs with a consistent structure.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Standard output contract for all agents.

    Attributes:
        source: Identifier for the producing agent (vision/text/barcode/fusion).
        confidence: Normalized confidence score in the range [0.0, 1.0].
        data: Flexible payload for extracted or synthesized meal details.
        error: Optional error message when processing fails.
    """

    source: str = Field(..., description="Agent source identifier")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    data: dict[str, Any] = Field(default_factory=dict, description="Agent payload")
    error: str | None = Field(default=None, description="Error details, if any")
