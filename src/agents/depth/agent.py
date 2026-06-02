"""Depth Anything based monocular depth generation agent."""

from __future__ import annotations

import asyncio
import io
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from agents.base_agent import BaseAgent
from schemas import AgentResponse, DepthAgentData, DepthInputSummary


class DepthAgent(BaseAgent):
    """Generates an 8-bit grayscale depth PNG from a single RGB image."""

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
    ) -> None:
        self.model_name = model_name
        self._pipeline = None

    async def process(self, input_data: Any) -> AgentResponse:
        payload: dict[str, Any] = {}
        try:
            payload = self._normalize_payload(input_data)
            depth_path = await asyncio.to_thread(
                self._generate_depth,
                payload["image_path"],
            )
            contract = DepthAgentData(
                input=DepthInputSummary(
                    image_filename=payload.get("filename"),
                    image_provided=True,
                ),
                model_name=self.model_name,
                depth_path=str(depth_path),
                llm_notes=[
                    "Depth was generated from RGB using monocular depth estimation.",
                    "Generated depth is relative, not metric camera depth.",
                    "Uploaded depth should be preferred when available.",
                ],
            )
            return AgentResponse(
                source="depth",
                confidence=0.65,
                data=contract.model_dump(),
                error=None,
            )
        except Exception as exc:
            return AgentResponse(
                source="depth",
                confidence=0.0,
                data={},
                error=f"Depth generation failed: {exc}",
            )
        finally:
            for temp_path in payload.get("_temp_paths", []):
                temp_path.unlink(missing_ok=True)

    def _normalize_payload(self, input_data: Any) -> dict[str, Any]:
        if isinstance(input_data, (str, Path)):
            path = Path(input_data)
            return {"image_path": path, "filename": path.name, "_temp_paths": []}

        if not isinstance(input_data, dict):
            raise TypeError("depth input must be an image path or dictionary payload")

        if input_data.get("image_path"):
            path = Path(input_data["image_path"])
            return {
                "image_path": path,
                "filename": input_data.get("image_filename") or input_data.get("filename") or path.name,
                "_temp_paths": [],
            }

        image_bytes = input_data.get("image_bytes") or input_data.get("bytes")
        if image_bytes is None:
            raise ValueError("depth payload must include image_path, image_bytes, or bytes")

        image_path = self._bytes_to_temp_image(image_bytes)
        return {
            "image_path": image_path,
            "filename": input_data.get("image_filename") or input_data.get("filename"),
            "_temp_paths": [image_path],
        }

    def _bytes_to_temp_image(self, image_bytes: bytes) -> Path:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            with tempfile.NamedTemporaryFile(
                prefix="depth_rgb_",
                suffix=".png",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
            image.save(temp_path)
        return temp_path

    def _generate_depth(self, image_path: Path) -> Path:
        pipe = self._load_pipeline()
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            result = pipe(rgb)

        depth = result.get("depth")
        if depth is None:
            raise RuntimeError("Depth Anything did not return a depth image.")

        depth_image = self._normalize_depth_image(depth)
        with tempfile.NamedTemporaryFile(
            prefix="generated_depth_",
            suffix=".png",
            delete=False,
        ) as temp_file:
            depth_path = Path(temp_file.name)
        depth_image.save(depth_path)
        return depth_path

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for DepthAgent. Install project requirements first."
            ) from exc

        self._pipeline = pipeline(
            task="depth-estimation",
            model=self.model_name,
        )
        return self._pipeline

    def _normalize_depth_image(self, depth: Any) -> Image.Image:
        if isinstance(depth, Image.Image):
            array = np.asarray(depth)
        else:
            array = np.asarray(depth)

        array = array.astype(np.float32)
        if array.ndim == 3:
            array = array[..., 0]

        finite = np.isfinite(array)
        if not finite.any():
            raise RuntimeError("Depth Anything returned non-finite depth values.")

        valid = array[finite]
        min_value = float(valid.min())
        max_value = float(valid.max())
        if max_value <= min_value:
            normalized = np.zeros_like(array, dtype=np.uint8)
        else:
            normalized = ((array - min_value) / (max_value - min_value) * 255.0)
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        return Image.fromarray(normalized, mode="L")
