from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import timm
from ultralytics import YOLO
from pydantic import BaseModel, Field

try:
    from .class_mapping import FOODSEG103_CLASSES
    from .nutrition_lookup import NutritionLookup
except ImportError:  # Allows running as a standalone script
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from agents.vision.class_mapping import FOODSEG103_CLASSES
    from agents.vision.nutrition_lookup import NutritionLookup

# 1. Pydantic Modelleri (ge=0.0 ile Sıfırın Altı Değerler Kesinlikle Yasaklandı)
class RegressionOutput(BaseModel):
    calories_kcal: float = Field(..., ge=0.0)
    mass_g: float = Field(..., ge=0.0)
    fat_g: float = Field(..., ge=0.0)
    carbs_g: float = Field(..., ge=0.0)
    protein_g: float = Field(..., ge=0.0)

class IngredientAllocation(BaseModel):
    name: str
    pixel_ratio: float = Field(..., ge=0.0)
    normalized_ratio: float = Field(..., ge=0.0)
    detection_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    mask_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    mass_g: float = Field(..., ge=0.0)
    calories_kcal: float = Field(..., ge=0.0)
    fat_g: float = Field(..., ge=0.0)
    carbs_g: float = Field(..., ge=0.0)
    protein_g: float = Field(..., ge=0.0)

class VisionFusionResult(BaseModel):
    image_path: str
    totals: RegressionOutput
    ratio_summary: dict[str, float]
    ingredients: list[IngredientAllocation]

# 2. PyTorch Mimari Sınıfları (Swin + Cross Attention - Dosyanla Uyumlu Bırakıldı)
class DualSwinBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True) -> None:
        super().__init__()
        self.rgb_backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.depth_backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_features = self.rgb_backbone(rgb)
        if depth.size(1) == 1:
            depth = depth.repeat(1, 3, 1, 1)
        depth_features = self.depth_backbone(depth)
        return rgb_features, depth_features

class NutritionModel(nn.Module):
    def __init__(self, backbone: DualSwinBackbone, feature_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)

        self.fc1 = nn.Linear(feature_dim * 2, feature_dim // 2)
        self.fc2 = nn.Linear(feature_dim // 2, feature_dim * 2)
        self.sigmoid = nn.Sigmoid()

        # The checkpoint has 5 output neurons, but the Colab fine-tuning loss
        # optimized only outputs[:, :2] for calories and mass. Runtime ignores
        # the remaining raw outputs and derives macros from nutrition priors.
        self.regression_head = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 5),
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        rgb_feat, depth_feat = self.backbone(rgb, depth)

        q = self.query_proj(rgb_feat)
        k = self.key_proj(depth_feat)
        v = self.value_proj(depth_feat)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (rgb_feat.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_features = torch.matmul(attention_weights, v)

        fused_feat = torch.cat((rgb_feat, attended_features), dim=1)

        ca_feat = F.relu(self.fc1(fused_feat))
        ca_weight = self.sigmoid(self.fc2(ca_feat))
        fused_feat = fused_feat * ca_weight

        return self.regression_head(fused_feat)

class RegressionModel:
    def __init__(self, checkpoint_path: Path, device: torch.device, model_name: str) -> None:
        self.device = device
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.model = self._load_model(checkpoint_path, model_name)
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, checkpoint_path: Path, model_name: str) -> nn.Module:
        backbone = DualSwinBackbone(model_name, pretrained=False)
        feature_dim = getattr(backbone.rgb_backbone, "num_features", 768)
        model = NutritionModel(backbone, feature_dim=feature_dim)
        
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state, strict=True)
        return model

    def predict(self, rgb: torch.Tensor, depth: torch.Tensor) -> RegressionOutput:
        with torch.no_grad():
            output = self.model(rgb.to(self.device), depth.to(self.device)).squeeze(0).cpu().numpy()

        output = np.asarray(output, dtype=np.float32).reshape(-1)
        
        # KRİTİK İYİLEŞTİRME 1: Çıktıların negatif olmasını engelleyen Kırpma (Clipping)
        output = np.maximum(output, 0.0)

        return RegressionOutput(
            calories_kcal=float(output[0]),
            mass_g=float(output[1]),
            fat_g=0.0,
            carbs_g=0.0,
            protein_g=0.0,
        )

class SegmentationModel:
    def __init__(self, weights_path: Path, conf: float = 0.25) -> None:
        self.model = YOLO(str(weights_path))
        self.conf = conf

    def predict(self, image_path: Path) -> list[dict[str, float | str]]:
        results = self.model.predict(str(image_path), conf=self.conf, verbose=False)
        result = results[0]
        if result.masks is None or result.masks.data is None:
            return []

        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        total_pixels = 0
        counts: dict[str, int] = {}
        weighted_confidence: dict[str, float] = {}
        image_h, image_w = result.orig_shape
        image_area = max(int(image_h) * int(image_w), 1)

        for idx, mask in enumerate(masks):
            pixel_count = int(np.sum(mask > 0.5))
            total_pixels += pixel_count
            class_id = int(classes[idx])
            raw_name = result.names[class_id]
            name = FOODSEG103_CLASSES.get(raw_name, raw_name)
            counts[name] = counts.get(name, 0) + pixel_count
            weighted_confidence[name] = weighted_confidence.get(name, 0.0) + (
                float(confidences[idx]) * pixel_count
            )

        items = []
        mask_coverage = min(total_pixels / image_area, 1.0)
        for name, pixel_count in counts.items():
            ratio = pixel_count / total_pixels if total_pixels else 0.0
            items.append({
                "name": name,
                "pixel_ratio": float(ratio),
                "pixel_count": int(pixel_count),
                "detection_confidence": (
                    float(weighted_confidence[name] / pixel_count)
                    if pixel_count
                    else None
                ),
                "mask_coverage": float(mask_coverage),
            })
        return items

class UnifiedVisionAgent:
    def __init__(
        self, yolo_weights: Path, regressor_checkpoint: Path, 
        image_size: int = 224, model_name: str = "swin_tiny_patch4_window7_224",
        device: torch.device | None = None, conf: float = 0.25
    ) -> None:
        self.image_size = image_size
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model_name = model_name
        self.segmentation = SegmentationModel(yolo_weights, conf=conf)
        self.regressor = RegressionModel(regressor_checkpoint, self.device, model_name=model_name)
        self.nutrition_lookup = NutritionLookup()

    def _load_rgb(self, image_path: Path) -> torch.Tensor:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
        rgb = TF.resize(rgb, (self.image_size, self.image_size), interpolation=Image.BILINEAR)
        rgb = TF.to_tensor(rgb)
        rgb = TF.normalize(rgb, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return rgb.unsqueeze(0)

    def _load_depth(self, depth_path: Path | None) -> torch.Tensor:
        if depth_path is None:
            raise ValueError("Depth image is required for the Swin RGB-D model.")
        with Image.open(depth_path) as img:
            depth = img.convert("L")
        depth = TF.resize(depth, (self.image_size, self.image_size), interpolation=Image.NEAREST)
        depth = TF.to_tensor(depth)
        return depth.unsqueeze(0)

    def _normalize_ratios(self, items: Iterable[dict]) -> list[dict]:
        ratio_sum = sum(float(item.get("pixel_ratio", 0.0)) for item in items)
        normalized = []
        for item in items:
            ratio = float(item.get("pixel_ratio", 0.0))
            normalized_ratio = ratio / ratio_sum if ratio_sum > 0 else 0.0
            normalized.append({
                "name": item["name"],
                "pixel_ratio": ratio,
                "normalized_ratio": normalized_ratio,
                "detection_confidence": item.get("detection_confidence"),
                "mask_coverage": item.get("mask_coverage"),
            })
        return normalized

    def _allocate(
        self,
        totals: RegressionOutput,
        items: list[dict],
    ) -> tuple[list[IngredientAllocation], dict[str, float]]:
        calculated = []
        for item in items:
            ratio = item["normalized_ratio"]
            mass_g = totals.mass_g * ratio
            prior = self.nutrition_lookup.lookup(str(item["name"]))
            calculated.append({
                "name": item["name"],
                "pixel_ratio": item["pixel_ratio"],
                "normalized_ratio": ratio,
                "detection_confidence": item.get("detection_confidence"),
                "mask_coverage": item.get("mask_coverage"),
                "mass_g": mass_g,
                "db_calories_kcal": mass_g * prior.calories_per_g,
                "fat_g": mass_g * prior.fat_per_g,
                "carbs_g": mass_g * prior.carbs_per_g,
                "protein_g": mass_g * prior.protein_per_g,
                "matched": prior.matched,
            })

        db_calories_total = sum(item["db_calories_kcal"] for item in calculated)
        calorie_scale = totals.calories_kcal / db_calories_total if db_calories_total > 0 else 0.0
        allocations = [
            IngredientAllocation(
                name=str(item["name"]),
                pixel_ratio=float(item["pixel_ratio"]),
                normalized_ratio=float(item["normalized_ratio"]),
                detection_confidence=(
                    float(item["detection_confidence"])
                    if item.get("detection_confidence") is not None
                    else None
                ),
                mask_coverage=(
                    float(item["mask_coverage"])
                    if item.get("mask_coverage") is not None
                    else None
                ),
                mass_g=float(item["mass_g"]),
                calories_kcal=float(item["db_calories_kcal"] * calorie_scale),
                fat_g=float(item["fat_g"]),
                carbs_g=float(item["carbs_g"]),
                protein_g=float(item["protein_g"]),
            )
            for item in calculated
        ]

        summary = {
            "db_calories_before_scaling_kcal": float(db_calories_total),
            "swin_calorie_anchor_kcal": float(totals.calories_kcal),
            "ingredient_calorie_scale": float(calorie_scale),
            "nutrition_prior_match_ratio": (
                sum(1 for item in calculated if item["matched"]) / len(calculated)
                if calculated
                else 0.0
            ),
        }
        return allocations, summary

    def run(self, image_path: Path, depth_path: Path | None) -> VisionFusionResult:
        items = self.segmentation.predict(image_path)
        
        # KRİTİK İYİLEŞTİRME 2: YOLO Sınıf Bulamazsa (Fallback Mechanism)
        if not items:
            items = [{
                "name": "Unknown Mixed Food",
                "pixel_ratio": 1.0,
                "pixel_count": 1
            }]

        normalized_items = self._normalize_ratios(items)

        rgb = self._load_rgb(image_path)
        depth = self._load_depth(depth_path)
        swin_totals = self.regressor.predict(rgb, depth)

        ratio_sum = sum(item["pixel_ratio"] for item in normalized_items)
        allocations, nutrition_summary = self._allocate(swin_totals, normalized_items)
        totals = RegressionOutput(
            calories_kcal=swin_totals.calories_kcal,
            mass_g=swin_totals.mass_g,
            fat_g=float(sum(item.fat_g for item in allocations)),
            carbs_g=float(sum(item.carbs_g for item in allocations)),
            protein_g=float(sum(item.protein_g for item in allocations)),
        )
        ratio_summary = {
            "raw_ratio_sum": float(ratio_sum),
            "normalized_ratio_sum": float(sum(item["normalized_ratio"] for item in normalized_items)),
            **nutrition_summary,
        }

        return VisionFusionResult(
            image_path=str(image_path),
            totals=totals,
            ratio_summary=ratio_summary,
            ingredients=allocations,
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO ingredient ratios + Swin-based nutrition regressor.")
    parser.add_argument("--image", type=Path, required=True, help="Path to RGB image.")
    parser.add_argument("--depth", type=Path, required=True, help="Path to depth image.")
    parser.add_argument("--yolo-weights", type=Path, required=True, help="Path to YOLOv8 segmentation weights.")
    parser.add_argument("--regressor-checkpoint", type=Path, required=True, help="Path to the Swin nutrition model checkpoint (.pth).")
    parser.add_argument("--model-name", default="swin_tiny_patch4_window7_224", help="Backbone name used during training.")
    parser.add_argument("--image-size", type=int, default=224, help="Resize size used by the regression model.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write JSON output.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.depth.exists():
        raise FileNotFoundError(f"Depth image not found: {args.depth}")
    if not args.yolo_weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {args.yolo_weights}")
    if not args.regressor_checkpoint.exists():
        raise FileNotFoundError(f"Regression checkpoint not found: {args.regressor_checkpoint}")

    agent = UnifiedVisionAgent(
        yolo_weights=args.yolo_weights,
        regressor_checkpoint=args.regressor_checkpoint,
        image_size=args.image_size,
        model_name=args.model_name,
        conf=args.conf,
    )
    result = agent.run(args.image, depth_path=args.depth)
    payload = result.model_dump()
    output_json = json.dumps(payload, indent=2)

    if args.output:
        args.output.write_text(output_json, encoding="utf-8")
        print(f"Saved: {args.output}")
    else:
        print(output_json)

if __name__ == "__main__":
    main()
