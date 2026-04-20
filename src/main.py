"""FastAPI entrypoint for multimodal meal analysis."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from agents.barcode.agent import BarcodeAgent
from agents.fusion.agent import FusionAgent
from agents.text.agent import TextAgent
from agents.vision.agent import VisionAgent
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

    vision_agent = VisionAgent()
    text_agent = TextAgent()
    barcode_agent = BarcodeAgent()
    fusion_agent = FusionAgent()

    # Prepare optional tasks so modalities can run concurrently.
    tasks: dict[str, asyncio.Task[AgentResponse]] = {}
    image_bytes: bytes | None = None
    image_filename: str | None = None
    barcode_image_bytes: bytes | None = None
    barcode_image_filename: str | None = None

    if image is not None:
        image_bytes = await image.read()
        image_filename = image.filename
        # Pass a minimal payload for now; replace with richer metadata as needed.
        image_payload: dict[str, Any] = {
            "filename": image.filename,
            "content_type": image.content_type,
            "bytes": image_bytes,
        }
        tasks["vision"] = asyncio.create_task(vision_agent.process(image_payload))

    if barcode_image is not None:
        barcode_image_bytes = await barcode_image.read()
        barcode_image_filename = barcode_image.filename

    if text:
        tasks["text"] = asyncio.create_task(text_agent.process(text))

    if barcode or barcode_image_bytes is not None:
        barcode_payload: dict[str, Any] = {
            "barcode": barcode,
            "image_bytes": barcode_image_bytes,
            "image_filename": barcode_image_filename,
        }
        tasks["barcode"] = asyncio.create_task(barcode_agent.process(barcode_payload))

    # Wait for all requested modality agents to complete.
    gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)

    agent_outputs: dict[str, AgentResponse | None] = {
        "vision": None,
        "text": None,
        "barcode": None,
    }

    # Map gathered results back to their modality keys in insertion order.
    for key, result in zip(tasks.keys(), gathered):
        if isinstance(result, Exception):
            agent_outputs[key] = AgentResponse(
                source=key,
                confidence=0.0,
                data={},
                error=f"{key} agent failed: {result}",
            )
        else:
            agent_outputs[key] = result

    # Fusion receives all modality outputs (including missing as None).
    fusion_result = await fusion_agent.process(agent_outputs)
    return fusion_result
