"""API-facing vision agent backed by the unified late-fusion vision pipeline."""

from __future__ import annotations

import asyncio
import io
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from agents.base_agent import BaseAgent
from schemas import (
    AgentResponse,
    NutritionEstimate,
    VisionAgentData,
    VisionDetectedItem,
    VisionIngredientEstimate,
    VisionInputSummary,
    VisionModelBundle,
)

from .unified_vision_agent import SegmentationModel, UnifiedVisionAgent


class VisionAgent(BaseAgent):
    """Runs YOLO segmentation and Swin nutrition regression as one vision agent.

    The full late-fusion path requires both an RGB image and a depth image. When
    depth is missing, the agent falls back to segmentation-only output so the
    existing API and GUI can still return visible food ratios.
    """

    def __init__(
        self,
        yolo_weights: str | Path | None = None,
        regressor_checkpoint: str | Path | None = None,
        confidence: float = 0.25,
        image_size: int = 224,
        model_name: str = "swin_tiny_patch4_window7_224",
    ) -> None:
        vision_dir = Path(__file__).resolve().parent
        self.yolo_weights = Path(yolo_weights) if yolo_weights else vision_dir / "best.pt"
        self.regressor_checkpoint = (
            Path(regressor_checkpoint)
            if regressor_checkpoint
            else vision_dir / "food101_pretrained_model_SwinV1-99acc-10epochfinetune.pth"
        )
        self.confidence = confidence
        self.image_size = image_size
        self.model_name = model_name
        self._pipeline: UnifiedVisionAgent | None = None
        self._segmentation: SegmentationModel | None = None

    @property
    def pipeline(self) -> UnifiedVisionAgent:
        if self._pipeline is None:
            self._pipeline = UnifiedVisionAgent(
                yolo_weights=self.yolo_weights,
                regressor_checkpoint=self.regressor_checkpoint,
                image_size=self.image_size,
                model_name=self.model_name,
                conf=self.confidence,
            )
        return self._pipeline

    @property
    def segmentation(self) -> SegmentationModel:
        if self._segmentation is None:
            self._segmentation = SegmentationModel(
                weights_path=self.yolo_weights,
                conf=self.confidence,
            )
        return self._segmentation

    async def process(self, input_data: Any) -> AgentResponse:
        """Run late-fusion vision inference for image/depth payloads."""
        payload: dict[str, Any] = {}
        try:
            payload = self._normalize_payload(input_data)
            image_path = payload["image_path"]
            depth_path = payload.get("depth_path")

            if depth_path is None:
                detected_items = await asyncio.to_thread(
                    self.segmentation.predict,
                    image_path,
                )
                contract = VisionAgentData(
                    mode="segmentation_only",
                    input=VisionInputSummary(
                        image_provided=True,
                        depth_provided=False,
                    ),
                    models=self._model_bundle("segmentation_only"),
                    detected_items=self._detected_item_contracts(detected_items),
                    llm_notes=[
                        "Only visible food segmentation was run because no depth image was provided.",
                        "Do not treat pixel ratios as grams without a separate mass or serving-size assumption.",
                    ],
                    missing_requirements=["depth_image_for_late_fusion"],
                )
                return AgentResponse(
                    source="vision",
                    confidence=self._estimate_confidence(
                        detected_items,
                        depth_provided=False,
                    ),
                    data=contract.model_dump(),
                    error=None if detected_items else "No food detected.",
                )

            result = await asyncio.to_thread(self.pipeline.run, image_path, depth_path)
            contract = VisionAgentData(
                mode="late_fusion",
                input=VisionInputSummary(
                    image_provided=True,
                    depth_provided=True,
                ),
                models=self._model_bundle("late_fusion"),
                detected_items=[
                    VisionDetectedItem(
                        name=ingredient.name,
                        pixel_ratio=round(ingredient.pixel_ratio, 4),
                        confidence=ingredient.detection_confidence,
                    )
                    for ingredient in result.ingredients
                ],
                totals=NutritionEstimate(
                    calories_kcal=result.totals.calories_kcal,
                    mass_g=result.totals.mass_g,
                    protein_g=result.totals.protein_g,
                    carbs_g=result.totals.carbs_g,
                    fat_g=result.totals.fat_g,
                ),
                ingredients=[
                    VisionIngredientEstimate(
                        name=ingredient.name,
                        pixel_ratio=ingredient.pixel_ratio,
                        normalized_ratio=ingredient.normalized_ratio,
                        confidence=ingredient.detection_confidence,
                        nutrition=NutritionEstimate(
                            calories_kcal=ingredient.calories_kcal,
                            mass_g=ingredient.mass_g,
                            protein_g=ingredient.protein_g,
                            carbs_g=ingredient.carbs_g,
                            fat_g=ingredient.fat_g,
                        ),
                    )
                    for ingredient in result.ingredients
                ],
                ratio_summary=result.ratio_summary,
                llm_notes=[
                    "Total calories and mass come from the Swin RGB-D regression model.",
                    "YOLO supplies visible ingredient classes and pixel ratios.",
                    "Protein, carbs, and fat are calculated from Nutrition5K nutrition priors using mass allocated by normalized YOLO pixel ratios.",
                    "Ingredient calories are scaled to the Swin total-calorie anchor.",
                ],
            )
            return AgentResponse(
                source="vision",
                confidence=self._estimate_confidence(
                    result.ingredients,
                    depth_provided=True,
                    ratio_summary=result.ratio_summary,
                ),
                data=contract.model_dump(),
                error=None,
            )
        except Exception as exc:
            return AgentResponse(
                source="vision",
                confidence=0.0,
                data={"detected_items": []},
                error=f"Vision analysis failed: {exc}",
            )
        finally:
            for temp_path in payload.get("_temp_paths", []):
                temp_path.unlink(missing_ok=True)

    def _normalize_payload(self, input_data: Any) -> dict[str, Any]:
        if isinstance(input_data, (str, Path)):
            return {"image_path": Path(input_data), "_temp_paths": []}

        if not isinstance(input_data, dict):
            raise TypeError("vision input must be an image path or dictionary payload")

        if input_data.get("image_path"):
            payload = {"image_path": Path(input_data["image_path"]), "_temp_paths": []}
        else:
            image_bytes = input_data.get("bytes") or input_data.get("image_bytes")
            if image_bytes is None:
                raise ValueError("vision payload must include image_path, bytes, or image_bytes")
            image_path = self._bytes_to_temp_image(image_bytes, "vision_rgb_")
            payload = {"image_path": image_path, "_temp_paths": [image_path]}

        if input_data.get("depth_path"):
            payload["depth_path"] = Path(input_data["depth_path"])
        else:
            depth_bytes = input_data.get("depth_bytes")
            if depth_bytes is not None:
                depth_path = self._bytes_to_temp_image(depth_bytes, "vision_depth_")
                payload["depth_path"] = depth_path
                payload["_temp_paths"].append(depth_path)

        return payload

    def _bytes_to_temp_image(self, image_bytes: bytes, prefix: str) -> Path:
        with Image.open(io.BytesIO(image_bytes)) as image:
            suffix = ".png"
            with tempfile.NamedTemporaryFile(
                prefix=prefix,
                suffix=suffix,
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
            image.save(temp_path)
        return temp_path

    def _detected_item_contracts(
        self,
        detected_items: list[dict[str, float | int | str]],
    ) -> list[VisionDetectedItem]:
        return [
            VisionDetectedItem(
                name=str(item["name"]),
                pixel_ratio=round(float(item.get("pixel_ratio", 0.0)), 4),
                confidence=(
                    float(item["detection_confidence"])
                    if item.get("detection_confidence") is not None
                    else None
                ),
            )
            for item in detected_items
        ]

    def _model_bundle(
        self,
        fusion_strategy: str,
    ) -> VisionModelBundle:
        return VisionModelBundle(
            segmentation_model=str(self.yolo_weights),
            nutrition_model=(
                str(self.regressor_checkpoint)
                if fusion_strategy == "late_fusion"
                else None
            ),
            fusion_strategy=fusion_strategy,
        )

    def _estimate_confidence(
        self,
        detected_items: Any,
        *,
        depth_provided: bool,
        ratio_summary: dict[str, Any] | None = None,
    ) -> float:
        items = list(detected_items or [])
        if not items:
            return 0.0

        detection_values = [
            value
            for item in items
            if (value := self._confidence_value(item)) is not None
        ]
        detection_score = (
            sum(detection_values) / len(detection_values)
            if detection_values
            else 0.45
        )

        coverage_values = [
            value
            for item in items
            if (value := self._coverage_value(item)) is not None
        ]
        coverage = max(coverage_values) if coverage_values else None
        if coverage is None:
            coverage_score = 0.65
        else:
            coverage_score = min(max(coverage / 0.35, 0.2), 1.0)

        prior_match = None
        if ratio_summary:
            prior_match = ratio_summary.get("nutrition_prior_match_ratio")
        prior_score = float(prior_match) if prior_match is not None else 0.7

        depth_score = 1.0 if depth_provided else 0.65

        confidence = (
            0.45 * detection_score
            + 0.20 * coverage_score
            + 0.20 * prior_score
            + 0.15 * depth_score
        )
        return round(min(max(confidence, 0.0), 0.95), 4)

    def _ratio_value(self, item: Any) -> float:
        if hasattr(item, "normalized_ratio"):
            return float(item.normalized_ratio)
        if hasattr(item, "pixel_ratio"):
            return float(item.pixel_ratio)
        if isinstance(item, dict):
            return float(item.get("normalized_ratio", item.get("pixel_ratio", 0.0)))
        return 0.0

    def _confidence_value(self, item: Any) -> float | None:
        if hasattr(item, "detection_confidence"):
            return item.detection_confidence
        if hasattr(item, "confidence"):
            return item.confidence
        if isinstance(item, dict):
            value = item.get("detection_confidence", item.get("confidence"))
            return float(value) if value is not None else None
        return None

    def _coverage_value(self, item: Any) -> float | None:
        if hasattr(item, "mask_coverage"):
            return item.mask_coverage
        if isinstance(item, dict):
            value = item.get("mask_coverage")
            return float(value) if value is not None else None
        return None
