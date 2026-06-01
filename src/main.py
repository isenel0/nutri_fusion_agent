"""FastAPI entrypoint for multimodal meal analysis."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from agents.decision.agent import FusionDecisionAgent
from agents.fusion.agent import FusionAgent
from agents.orchestrator.agent import OrchestratorAgent
from schemas import AgentResponse

app = FastAPI(
    title="Nutri Fusion Agent API",
    version="0.1.0",
    description="Boilerplate backend for multimodal dietary tracking.",
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Simple service health endpoint."""
    return {"status": "ok"}


@app.post("/analyze-meal", response_model=AgentResponse)
async def analyze_meal(
    image: UploadFile | None = File(default=None),
    depth_image: UploadFile | None = File(default=None),
    barcode_image: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    barcode: str | None = Form(default=None),
) -> AgentResponse:
    """Analyze meal input from one or more modalities.

    This endpoint accepts optional image, text, and barcode inputs. It invokes
    each available modality agent, then passes their outputs to the FusionAgent
    to produce a single reconciled response.
    """

    if image is None and barcode_image is None and not text and not barcode:
        raise HTTPException(
            status_code=400,
            detail=(
                "Provide at least one input: image, barcode_image, text, or barcode."
            ),
        )

    orchestrator = OrchestratorAgent()
    decision_agent = FusionDecisionAgent()
    fusion_agent = FusionAgent()

    payload: dict[str, Any] = {
        "text": text,
        "barcode": barcode,
    }

    if image is not None:
        payload["image_bytes"] = await image.read()
        payload["image_filename"] = image.filename
        payload["image_content_type"] = image.content_type

    if depth_image is not None:
        payload["depth_bytes"] = await depth_image.read()
        payload["depth_image_filename"] = depth_image.filename

    if barcode_image is not None:
        payload["barcode_image_bytes"] = await barcode_image.read()
        payload["barcode_image_filename"] = barcode_image.filename

    orchestrator_result = await orchestrator.process(payload)
    decision_result = await decision_agent.process(orchestrator_result)
    fusion_result = await fusion_agent.process(
        {
            "orchestrator": orchestrator_result,
            "decision": decision_result,
        }
    )
    return fusion_result
