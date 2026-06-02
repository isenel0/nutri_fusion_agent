"""Shared Pydantic schemas used across all agents.

Every agent in the pipeline should return an AgentResponse so the
fusion layer can consume outputs with a consistent structure.
"""

from __future__ import annotations

from typing import Any, Literal

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


class NutritionEstimate(BaseModel):
    """Common nutrition estimate fields used by multiple agents."""

    calories_kcal: float | None = Field(default=None, ge=0.0)
    mass_g: float | None = Field(default=None, ge=0.0)
    protein_g: float | None = Field(default=None, ge=0.0)
    carbs_g: float | None = Field(default=None, ge=0.0)
    fat_g: float | None = Field(default=None, ge=0.0)
    fiber_g: float | None = Field(default=None, ge=0.0)
    sugar_g: float | None = Field(default=None, ge=0.0)
    sodium_g: float | None = Field(default=None, ge=0.0)


class DepthInputSummary(BaseModel):
    """Input used by the depth generation agent."""

    image_filename: str | None = None
    image_provided: bool = True


class DepthAgentData(BaseModel):
    """Structured DepthAgent payload for generated monocular depth."""

    contract_version: Literal["depth.v1"] = "depth.v1"
    agent: Literal["depth"] = "depth"
    input: DepthInputSummary
    mode: Literal["generated_from_rgb"] = "generated_from_rgb"
    model_name: str
    depth_path: str
    output_format: Literal["8bit_grayscale_png"] = "8bit_grayscale_png"
    llm_notes: list[str] = Field(default_factory=list)


class VisionModelBundle(BaseModel):
    """Model bundle used by the vision agent."""

    segmentation_model: str = Field(..., description="YOLO segmentation model path/name")
    nutrition_model: str | None = Field(
        default=None,
        description="Swin nutrition regression checkpoint path/name",
    )
    fusion_strategy: Literal["late_fusion", "segmentation_only"]


class VisionInputSummary(BaseModel):
    """Input availability summary for LLM orchestration."""

    image_provided: bool
    depth_provided: bool


class VisionDetectedItem(BaseModel):
    """Visible item detected by the segmentation model."""

    name: str
    pixel_ratio: float = Field(..., ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class VisionIngredientEstimate(BaseModel):
    """Per-ingredient allocation after late fusion."""

    name: str
    pixel_ratio: float = Field(..., ge=0.0, le=1.0)
    normalized_ratio: float = Field(..., ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    nutrition: NutritionEstimate


class VisionAgentData(BaseModel):
    """Structured VisionAgent payload for the LLM orchestrator."""

    contract_version: Literal["vision.v1"] = "vision.v1"
    agent: Literal["vision"] = "vision"
    mode: Literal["late_fusion", "segmentation_only"]
    input: VisionInputSummary
    models: VisionModelBundle
    detected_items: list[VisionDetectedItem] = Field(default_factory=list)
    totals: NutritionEstimate | None = None
    ingredients: list[VisionIngredientEstimate] = Field(default_factory=list)
    ratio_summary: dict[str, float] = Field(default_factory=dict)
    llm_notes: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)


class BarcodeInputSummary(BaseModel):
    """Input source summary for barcode resolution."""

    barcode_text: str | None = None
    image_filename: str | None = None


class BarcodeIdentity(BaseModel):
    """Resolved barcode identity."""

    value: str
    type: str | None = None


class BarcodeProduct(BaseModel):
    """Packaged-product metadata from the nutrition provider."""

    name: str | None = None
    brand: str | None = None
    categories: str | None = None
    quantity: str | None = None
    serving_size: str | None = None


class BarcodeSource(BaseModel):
    """Barcode data provenance."""

    provider: str = "openfoodfacts"
    fetched_barcode: str | None = None


class BarcodeAgentData(BaseModel):
    """Structured BarcodeAgent payload for the LLM orchestrator."""

    contract_version: Literal["barcode.v1"] = "barcode.v1"
    agent: Literal["barcode"] = "barcode"
    input: BarcodeInputSummary
    barcode: BarcodeIdentity | None = None
    product: BarcodeProduct | None = None
    nutrition_per_100g: NutritionEstimate | None = None
    source: BarcodeSource = Field(default_factory=BarcodeSource)
    llm_notes: list[str] = Field(default_factory=list)
    raw_product: dict[str, Any] | None = None


class TextInputSummary(BaseModel):
    """Raw text received by the text agent."""

    raw_text: str


class TextEntity(BaseModel):
    """Entity extracted by the local text token-classification model."""

    type: Literal["food", "quantity", "unit", "method", "modifier"]
    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)


class TextMealParse(BaseModel):
    """Convenience grouping for deterministic fusion and LLM orchestration."""

    foods: list[str] = Field(default_factory=list)
    quantities: list[str] = Field(default_factory=list)
    units: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    modifiers: list[str] = Field(default_factory=list)
    portion_multiplier: float | None = Field(default=None, ge=0.0)


class TextModelInfo(BaseModel):
    """Local text model metadata."""

    model_path: str
    architecture: str = "BertForTokenClassification"
    labels: list[str] = Field(default_factory=list)


class TextAgentData(BaseModel):
    """Structured TextAgent payload for the LLM orchestrator."""

    contract_version: Literal["text.v1"] = "text.v1"
    agent: Literal["text"] = "text"
    input: TextInputSummary
    model: TextModelInfo
    entities: list[TextEntity] = Field(default_factory=list)
    parsed: TextMealParse = Field(default_factory=TextMealParse)
    llm_notes: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)


class TextNutritionInputSummary(BaseModel):
    """Inputs used by text-based nutrition resolution."""

    raw_text: str
    vision_mass_g: float | None = Field(default=None, ge=0.0)


class TextNutritionItemEstimate(BaseModel):
    """Per-food nutrition estimate derived from text and nutrition priors."""

    name: str
    lookup_name: str
    canonicalization_source: Literal["llm", "alias_fallback", "raw"] = "raw"
    mass_g: float | None = Field(default=None, ge=0.0)
    mass_source: Literal["explicit_text", "vision_distributed", "default_serving", "unknown"]
    nutrition: NutritionEstimate
    confidence: float = Field(..., ge=0.0, le=1.0)
    assumptions: list[str] = Field(default_factory=list)


class TextNutritionAgentData(BaseModel):
    """Structured text nutrition payload for fusion."""

    contract_version: Literal["text_nutrition.v1"] = "text_nutrition.v1"
    agent: Literal["text_nutrition"] = "text_nutrition"
    input: TextNutritionInputSummary
    items: list[TextNutritionItemEstimate] = Field(default_factory=list)
    totals: NutritionEstimate | None = None
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    llm_notes: list[str] = Field(default_factory=list)


class IngredientSourceLink(BaseModel):
    """Source evidence linked to a resolved ingredient."""

    source: Literal["vision", "text", "text_nutrition", "barcode"]
    name: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    mass_g: float | None = Field(default=None, ge=0.0)
    action: Literal["match", "replace", "add", "preserve"]


class ResolvedIngredient(BaseModel):
    """Canonical ingredient selected after cross-agent identity resolution."""

    display_name: str
    canonical_name: str
    source: Literal["vision", "text_nutrition", "barcode", "mixed"]
    mass_g: float | None = Field(default=None, ge=0.0)
    mass_source: Literal["vision", "explicit_text", "vision_remaining", "default_serving", "barcode", "unknown"]
    nutrition: NutritionEstimate
    confidence: float = Field(..., ge=0.0, le=1.0)
    links: list[IngredientSourceLink] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class IngredientResolutionAgentData(BaseModel):
    """Resolved ingredient identities before final macro fusion."""

    contract_version: Literal["ingredient_resolution.v1"] = "ingredient_resolution.v1"
    agent: Literal["ingredient_resolution"] = "ingredient_resolution"
    items: list[ResolvedIngredient] = Field(default_factory=list)
    totals: NutritionEstimate | None = None
    replacements: dict[str, str] = Field(default_factory=dict)
    unmatched_vision: list[str] = Field(default_factory=list)
    unmatched_text: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    llm_notes: list[str] = Field(default_factory=list)


class OrchestratorInputSummary(BaseModel):
    """Raw modality availability received by the orchestrator."""

    image_provided: bool = False
    depth_image_provided: bool = False
    text_provided: bool = False
    barcode_text_provided: bool = False
    barcode_image_provided: bool = False


class OrchestratorAgentData(BaseModel):
    """Structured orchestrator payload before final fusion."""

    contract_version: Literal["orchestrator.v1"] = "orchestrator.v1"
    agent: Literal["orchestrator"] = "orchestrator"
    input: OrchestratorInputSummary
    tool_order: list[str] = Field(default_factory=list)
    outputs: dict[str, AgentResponse | None] = Field(default_factory=dict)
    errors: dict[str, str] = Field(default_factory=dict)
    llm_context: dict[str, Any] = Field(default_factory=dict)
    llm_notes: list[str] = Field(default_factory=list)


class MealItemResult(BaseModel):
    """Final per-item estimate after fusion."""

    name: str
    source: Literal["vision", "barcode", "text", "text_nutrition", "fallback"]
    mass_g: float | None = Field(default=None, ge=0.0)
    nutrition: NutritionEstimate
    confidence: float = Field(..., ge=0.0, le=1.0)
    notes: list[str] = Field(default_factory=list)


class FusionDecision(BaseModel):
    """Deterministic fusion choices used to build the final meal result."""

    primary_nutrition_source: Literal["barcode", "vision", "text_nutrition", "ingredient_resolution", "fallback"]
    mass_source: Literal["vision", "text", "assumed_100g", "unknown"]
    portion_multiplier: float = Field(default=1.0, ge=0.0)
    corrected_food_names: dict[str, str] = Field(default_factory=dict)
    hidden_ingredients: list[str] = Field(default_factory=list)
    applied_rules: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    explanation: str | None = None


class MealAnalysisResult(BaseModel):
    """Final fused meal analysis contract."""

    contract_version: Literal["meal_analysis.v1"] = "meal_analysis.v1"
    agent: Literal["fusion"] = "fusion"
    meal_name: str | None = None
    final_macros: NutritionEstimate
    calculation_ingredients: list[MealItemResult] = Field(
        default_factory=list,
        description="Ingredient or product entries whose nutrition values were summed into final_macros.",
    )
    items: list[MealItemResult] = Field(default_factory=list)
    fusion_decision: FusionDecision
    inputs_used: dict[str, bool] = Field(default_factory=dict)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning_summary: str
    warnings: list[str] = Field(default_factory=list)
    agent_outputs: dict[str, AgentResponse | None] = Field(default_factory=dict)
