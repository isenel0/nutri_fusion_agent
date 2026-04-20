"""Vision agent baseline with EfficientNet-B0."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import torch
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from agents.base_agent import BaseAgent
from schemas import AgentResponse


class VisionAgent(BaseAgent):
    """Simple image classifier using EfficientNet-B0."""

    def __init__(self) -> None:
        self.weights = EfficientNet_B0_Weights.DEFAULT
        self.model = efficientnet_b0(weights=self.weights)
        self.model.eval()

        self.preprocess = self.weights.transforms()
        self.labels = self.weights.meta["categories"]

    async def process(self, input_data: Any) -> AgentResponse:
        """Run one forward pass and return top-1 prediction."""
        try:
            image = self._load_image(input_data)

            tensor = self.preprocess(image).unsqueeze(0)

            with torch.inference_mode():
                logits = self.model(tensor)
                probs = torch.softmax(logits[0], dim=0)
                confidence, index = torch.max(probs, dim=0)

            class_index = int(index.item())
            predicted_label = self.labels[class_index]

            return AgentResponse(
                source="vision",
                confidence=float(confidence.item()),
                data={
                    "model": "efficientnet_b0",
                    "predicted_label": predicted_label,
                },
                error=None,
            )
        except Exception as exc:
            return AgentResponse(
                source="vision",
                confidence=0.0,
                data={"model": "efficientnet_b0"},
                error=f"Vision inference failed: {exc}",
            )

    def _load_image(self, input_data: Any) -> Image.Image:
        """Accept either image bytes dict or image path string."""
        if isinstance(input_data, str):
            return Image.open(input_data).convert("RGB")

        if isinstance(input_data, dict):
            image_bytes = input_data.get("bytes") or input_data.get("image_bytes")
            image_path = input_data.get("image_path")

            if image_bytes is not None:
                return Image.open(BytesIO(image_bytes)).convert("RGB")

            if image_path:
                return Image.open(image_path).convert("RGB")

        raise ValueError("Expected image path or payload with image bytes.")
