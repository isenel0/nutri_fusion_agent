"""Evaluate the multimodal meal pipeline on local sample metadata.

The evaluator reads sample folders containing:

    sample_0001/
      rgb.jpg
      metadata.json

It runs image/text combinations, stores raw outputs, compares final predictions
against metadata ground truth, and exports per-case and aggregate metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agents.decision.agent import FusionDecisionAgent  # noqa: E402
from agents.fusion.agent import FusionAgent  # noqa: E402
from agents.orchestrator.agent import OrchestratorAgent  # noqa: E402
from agents.orchestrator.tools import (  # noqa: E402
    analyze_text,
    analyze_text_nutrition,
    resolve_ingredients,
)
from schemas import AgentResponse, OrchestratorAgentData, OrchestratorInputSummary  # noqa: E402


MACRO_KEYS = ("calories_kcal", "mass_g", "protein_g", "carbs_g", "fat_g")
SUPPORTED_RGB_FORMATS = {"JPEG", "JPG", "PNG", "MPO"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation over evaluation_dataset samples."
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=ROOT.parent / "evaluation_dataset" / "dataset" / "samples",
        help="Directory containing sample_*/metadata.json folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "system_eval",
        help="Directory where evaluation outputs are written.",
    )
    parser.add_argument(
        "--variants",
        default="A,B,C,D,E,F",
        help="Comma-separated text input variants to evaluate.",
    )
    parser.add_argument(
        "--modes",
        default="image_text",
        help=(
            "Comma-separated modes. Supported: image_text, text_only. "
            "image_text uses RGB plus each text variant; text_only uses text without image."
        ),
    )
    parser.add_argument("--start", default=None, help="First sample id, e.g. sample_0002.")
    parser.add_argument("--end", default=None, help="Last sample id, e.g. sample_0030.")
    parser.add_argument(
        "--sample-ids",
        default=None,
        help="Optional comma-separated exact sample ids to evaluate, e.g. sample_0002,sample_0004.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max sample count.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a case if its raw output JSON already exists.",
    )
    parser.add_argument(
        "--no-image-cache",
        action="store_true",
        help="Run the complete orchestrator for every image_text case instead of reusing image evidence.",
    )
    parser.add_argument(
        "--no-barcode",
        action="store_true",
        help="Do not pass metadata barcode values or barcode images into the pipeline.",
    )
    parser.add_argument(
        "--barcode-source",
        choices=("auto", "none", "text", "image", "both"),
        default="auto",
        help=(
            "Barcode evidence to pass into the pipeline. auto uses metadata barcode text "
            "and barcode image when available; none disables barcode evidence."
        ),
    )
    parser.add_argument(
        "--allow-invalid-images",
        action="store_true",
        help="Continue even if selected image_text samples contain unreadable RGB files.",
    )
    return parser.parse_args()


def load_samples(
    samples_dir: Path,
    start: str | None,
    end: str | None,
    limit: int | None,
    sample_ids: str | None = None,
) -> list[Path]:
    sample_dirs = sorted(
        path for path in samples_dir.glob("sample_*") if (path / "metadata.json").exists()
    )
    if sample_ids:
        requested = {item.strip() for item in sample_ids.split(",") if item.strip()}
        sample_dirs = [path for path in sample_dirs if path.name in requested]
    if start:
        sample_dirs = [path for path in sample_dirs if path.name >= start]
    if end:
        sample_dirs = [path for path in sample_dirs if path.name <= end]
    if limit is not None:
        sample_dirs = sample_dirs[:limit]
    return sample_dirs


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def validate_rgb_image(path: Path) -> str | None:
    if not path.exists():
        return f"Image file not found: {path}"
    try:
        with Image.open(path) as image:
            image_format = (image.format or "").upper()
            if image_format not in SUPPORTED_RGB_FORMATS:
                return (
                    f"Unsupported image format for vision pipeline: {path} "
                    f"has format {image.format!r}. Convert it to JPEG or PNG before evaluation."
                )
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return (
            f"Unsupported or unreadable image: {path}. "
            "Convert HEIC/HEIF inputs to real JPEG or PNG before evaluation. "
            f"Reader error: {exc}"
        )
    return None


def preflight_images(sample_dirs: list[Path], output_dir: Path) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for sample_dir in sample_dirs:
        metadata_path = sample_dir / "metadata.json"
        metadata = read_json(metadata_path)
        rgb_name = (metadata.get("images") or {}).get("rgb") or "rgb.jpg"
        rgb_path = sample_dir / rgb_name
        error = validate_rgb_image(rgb_path)
        if error:
            errors.append(
                {
                    "sample_id": str(metadata.get("sample_id") or sample_dir.name),
                    "image_path": str(rgb_path),
                    "error": error,
                }
            )
    if errors:
        write_json(output_dir / "preflight_image_errors.json", errors)
    return errors


async def run_full_pipeline(payload: dict[str, Any]) -> dict[str, AgentResponse]:
    orchestrator = OrchestratorAgent()
    decision_agent = FusionDecisionAgent()
    fusion_agent = FusionAgent()

    orchestrator_result = await orchestrator.process(payload)
    decision_result = await decision_agent.process(orchestrator_result)
    fusion_result = await fusion_agent.process(
        {
            "orchestrator": orchestrator_result,
            "decision": decision_result,
        }
    )
    return {
        "orchestrator": orchestrator_result,
        "decision": decision_result,
        "fusion": fusion_result,
    }


async def run_cached_image_text_case(
    image_orchestrator: AgentResponse,
    text: str | None,
    barcode_present: bool,
    barcode_image_present: bool = False,
) -> dict[str, AgentResponse]:
    orchestrator = OrchestratorAgent()
    decision_agent = FusionDecisionAgent()
    fusion_agent = FusionAgent()

    image_outputs = image_orchestrator.data.get("outputs") or {}
    outputs = {
        key: coerce_response(value)
        for key, value in deepcopy(image_outputs).items()
    }
    errors = dict(image_orchestrator.data.get("errors") or {})
    tool_order = list(image_orchestrator.data.get("tool_order") or [])

    if text:
        text_result = await analyze_text({"text": text})
        outputs["text"] = text_result
        tool_order.append("text")
        if text_result.error:
            errors["text"] = text_result.error

        text_nutrition_result = await analyze_text_nutrition(
            {
                "text": text_result,
                "vision": coerce_response(outputs.get("vision")),
            }
        )
        outputs["text_nutrition"] = text_nutrition_result
        tool_order.append("text_nutrition")
        if text_nutrition_result.error:
            errors["text_nutrition"] = text_nutrition_result.error

        ingredient_resolution_result = await resolve_ingredients(
            {
                "vision": coerce_response(outputs.get("vision")),
                "text_nutrition": text_nutrition_result,
            }
        )
        outputs["ingredient_resolution"] = ingredient_resolution_result
        tool_order.append("ingredient_resolution")
        if ingredient_resolution_result.error:
            errors["ingredient_resolution"] = ingredient_resolution_result.error

    input_summary = OrchestratorInputSummary(
        image_provided=bool(outputs.get("vision")),
        depth_image_provided=bool(outputs.get("depth")),
        text_provided=bool(text),
        barcode_text_provided=barcode_present,
        barcode_image_provided=barcode_image_present,
    )
    contract = OrchestratorAgentData(
        input=input_summary,
        tool_order=dedupe(tool_order),
        outputs=outputs,
        errors=errors,
        llm_context=orchestrator._llm_context(outputs),
        llm_notes=[
            "Evaluation bundle composed from cached image evidence and current text variant.",
        ],
    )
    orchestrator_result = AgentResponse(
        source="orchestrator",
        confidence=orchestrator._bundle_confidence(outputs),
        data=contract.model_dump(),
        error=None if outputs else "No modality inputs were provided.",
    )
    decision_result = await decision_agent.process(orchestrator_result)
    fusion_result = await fusion_agent.process(
        {
            "orchestrator": orchestrator_result,
            "decision": decision_result,
        }
    )
    return {
        "orchestrator": orchestrator_result,
        "decision": decision_result,
        "fusion": fusion_result,
    }


def coerce_response(value: Any) -> AgentResponse | None:
    if value is None:
        return None
    if isinstance(value, AgentResponse):
        return value
    if isinstance(value, dict):
        return AgentResponse.model_validate(value)
    return None


def dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def build_payload(
    sample_dir: Path,
    metadata: dict[str, Any],
    text: str | None,
    include_image: bool,
    barcode_source: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"text": text}
    if include_image:
        rgb_name = (metadata.get("images") or {}).get("rgb") or "rgb.jpg"
        payload["image_path"] = str(sample_dir / rgb_name)
    images = metadata.get("images") or {}
    barcode_image_name = images.get("barcode")
    if barcode_source in {"auto", "text", "both"} and metadata.get("barcode"):
        payload["barcode"] = metadata.get("barcode")
    if barcode_source in {"auto", "image", "both"} and barcode_image_name:
        payload["barcode_image_path"] = str(sample_dir / barcode_image_name)
    return payload


def barcode_enabled(metadata: dict[str, Any], barcode_source: str) -> bool:
    if barcode_source == "none":
        return False
    images = metadata.get("images") or {}
    has_text = bool(metadata.get("barcode"))
    has_image = bool(images.get("barcode"))
    if barcode_source == "text":
        return has_text
    if barcode_source == "image":
        return has_image
    if barcode_source == "both":
        return has_text or has_image
    return has_text or has_image


def final_data(outputs: dict[str, AgentResponse]) -> dict[str, Any]:
    fusion = outputs["fusion"]
    return fusion.data if isinstance(fusion.data, dict) else {}


def extract_prediction(outputs: dict[str, AgentResponse]) -> dict[str, Any]:
    data = final_data(outputs)
    macros = data.get("final_macros") or {}
    return {key: to_float(macros.get(key)) for key in MACRO_KEYS}


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def case_metrics(
    sample_id: str,
    variant_key: str,
    variant_name: str,
    mode: str,
    barcode_source: str,
    barcode_evidence_available: bool,
    text: str | None,
    metadata: dict[str, Any],
    outputs: dict[str, AgentResponse] | None,
    error: str | None,
) -> dict[str, Any]:
    gt = ((metadata.get("ground_truth") or {}).get("totals") or {})
    pred = extract_prediction(outputs) if outputs else {key: None for key in MACRO_KEYS}
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "mode": mode,
        "variant": variant_key,
        "variant_name": variant_name,
        "barcode_source": barcode_source,
        "barcode_evidence_available": barcode_evidence_available,
        "text_input": text or "",
        "error": error or "",
    }
    for key in MACRO_KEYS:
        gt_value = to_float(gt.get(key))
        pred_value = pred.get(key)
        row[f"gt_{key}"] = gt_value
        row[f"pred_{key}"] = pred_value
        row[f"abs_error_{key}"] = (
            abs(pred_value - gt_value)
            if pred_value is not None and gt_value is not None
            else None
        )
        row[f"signed_error_{key}"] = (
            pred_value - gt_value
            if pred_value is not None and gt_value is not None
            else None
        )

    ingredient_scores = ingredient_metrics(metadata, final_data(outputs) if outputs else {})
    row.update(ingredient_scores)

    if outputs:
        fusion_data = final_data(outputs)
        decision = fusion_data.get("fusion_decision") or {}
        row["confidence"] = fusion_data.get("confidence")
        row["primary_source"] = decision.get("primary_nutrition_source")
        row["mass_source"] = decision.get("mass_source")
        row["inputs_used"] = json.dumps(fusion_data.get("inputs_used") or {}, ensure_ascii=False)
        row["fusion_error"] = outputs["fusion"].error or ""
    else:
        row["confidence"] = None
        row["primary_source"] = None
        row["mass_source"] = None
        row["inputs_used"] = "{}"
        row["fusion_error"] = ""
    return row


def ingredient_metrics(metadata: dict[str, Any], fusion_data: dict[str, Any]) -> dict[str, Any]:
    gt_items = ((metadata.get("ground_truth") or {}).get("ingredients") or [])
    gt_names = []
    for item in gt_items:
        if not isinstance(item, dict):
            continue
        names = [
            normalize_name(item.get("canonical_name")),
            normalize_name(item.get("name_tr")),
        ]
        gt_names.append({name for name in names if name})

    pred_items = fusion_data.get("calculation_ingredients") or fusion_data.get("items") or []
    pred_names = [
        normalize_name(item.get("name"))
        for item in pred_items
        if isinstance(item, dict) and normalize_name(item.get("name"))
    ]

    matched_gt: set[int] = set()
    true_positive = 0
    for pred_name in pred_names:
        for index, aliases in enumerate(gt_names):
            if index in matched_gt:
                continue
            if any(name_matches(pred_name, gt_name) for gt_name in aliases):
                matched_gt.add(index)
                true_positive += 1
                break

    false_positive = max(len(pred_names) - true_positive, 0)
    false_negative = max(len(gt_names) - true_positive, 0)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "gt_ingredient_count": len(gt_names),
        "pred_ingredient_count": len(pred_names),
        "ingredient_tp": true_positive,
        "ingredient_fp": false_positive,
        "ingredient_fn": false_negative,
        "ingredient_precision": precision,
        "ingredient_recall": recall,
        "ingredient_f1": f1,
        "predicted_ingredients": "; ".join(pred_names),
    }


def normalize_name(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9çğıöşüı ]+", " ", text)).strip()


def name_matches(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return left == right or left in right or right in left


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (
                str(row["mode"]),
                str(row["variant"]),
                str(row.get("barcode_source") or ""),
            ),
            [],
        ).append(row)
    groups.setdefault(("ALL", "ALL", "ALL"), rows)

    summary: list[dict[str, Any]] = []
    for (mode, variant, barcode_source), group_rows in sorted(groups.items()):
        record: dict[str, Any] = {
            "mode": mode,
            "variant": variant,
            "barcode_source": barcode_source,
            "n_cases": len(group_rows),
            "n_success": sum(1 for row in group_rows if not row.get("error")),
        }
        for key in MACRO_KEYS:
            abs_values = numeric_values(row.get(f"abs_error_{key}") for row in group_rows)
            signed_values = numeric_values(row.get(f"signed_error_{key}") for row in group_rows)
            record[f"mae_{key}"] = mean(abs_values)
            record[f"rmse_{key}"] = rmse(signed_values)
            record[f"mean_signed_error_{key}"] = mean(signed_values)
        for key in ("ingredient_precision", "ingredient_recall", "ingredient_f1"):
            record[f"mean_{key}"] = mean(numeric_values(row.get(key) for row in group_rows))
        summary.append(record)
    return summary


def numeric_values(values: Any) -> list[float]:
    result = []
    for value in values:
        if value is None or value == "":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def rmse(values: list[float]) -> float | None:
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def persist_results(output_dir: Path, rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    write_csv(output_dir / "case_results.csv", rows)
    write_csv(output_dir / "metrics_summary.csv", aggregate(rows))
    write_json(output_dir / "failures.json", failures)


async def evaluate(args: argparse.Namespace) -> Path:
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    barcode_source = "none" if args.no_barcode else args.barcode_source
    unsupported_modes = set(modes) - {"image_text", "text_only"}
    if unsupported_modes:
        raise ValueError(f"Unsupported modes: {sorted(unsupported_modes)}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"system_eval_{run_id}"
    raw_dir = output_dir / "raw"
    write_json(
        output_dir / "config.json",
        {
            "samples_dir": str(args.samples_dir),
            "variants": variants,
            "modes": modes,
            "start": args.start,
            "end": args.end,
            "sample_ids": args.sample_ids,
            "limit": args.limit,
            "image_cache": not args.no_image_cache,
            "barcode_source": barcode_source,
        },
    )

    sample_dirs = load_samples(
        args.samples_dir,
        args.start,
        args.end,
        args.limit,
        args.sample_ids,
    )
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    if "image_text" in modes:
        image_errors = preflight_images(sample_dirs, output_dir)
        if image_errors and not args.allow_invalid_images:
            persist_results(output_dir, rows, failures)
            examples = "\n".join(
                f"- {item['sample_id']}: {item['image_path']}" for item in image_errors[:10]
            )
            raise RuntimeError(
                "Image preflight failed for selected samples. "
                f"See {output_dir / 'preflight_image_errors.json'}.\n{examples}"
            )

    try:
        for sample_dir in sample_dirs:
            metadata = read_json(sample_dir / "metadata.json")
            sample_id = str(metadata.get("sample_id") or sample_dir.name)
            print(f"\n[{sample_id}]")

            image_orchestrator: AgentResponse | None = None
            if "image_text" in modes and not args.no_image_cache:
                try:
                    image_payload = build_payload(
                        sample_dir=sample_dir,
                        metadata=metadata,
                        text=None,
                        include_image=True,
                        barcode_source=barcode_source,
                    )
                    image_orchestrator = (await run_full_pipeline(image_payload))["orchestrator"]
                    write_json(
                        raw_dir / sample_id / "_cached_image_orchestrator.json",
                        image_orchestrator.model_dump(),
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "sample_id": sample_id,
                            "mode": "image_cache",
                            "variant": "",
                            "error": str(exc),
                        }
                    )

            for mode in modes:
                for variant_key in variants:
                    text_item = (metadata.get("text_inputs") or {}).get(variant_key) or {}
                    variant_name = str(text_item.get("version_name") or variant_key)
                    text = text_item.get("text_input")
                    case_name = f"{mode}_{variant_key}_{variant_name}"
                    raw_path = raw_dir / sample_id / f"{case_name}.json"
                    if args.skip_existing and raw_path.exists():
                        continue
                    print(f"  - {case_name}")

                    outputs: dict[str, AgentResponse] | None = None
                    error: str | None = None
                    try:
                        if mode == "image_text":
                            if image_orchestrator is not None:
                                outputs = await run_cached_image_text_case(
                                    image_orchestrator=image_orchestrator,
                                    text=text,
                                    barcode_present=(
                                        barcode_source in {"auto", "text", "both"}
                                        and bool(metadata.get("barcode"))
                                    ),
                                    barcode_image_present=(
                                        barcode_source in {"auto", "image", "both"}
                                        and bool((metadata.get("images") or {}).get("barcode"))
                                    ),
                                )
                            else:
                                payload = build_payload(
                                    sample_dir=sample_dir,
                                    metadata=metadata,
                                    text=text,
                                    include_image=True,
                                    barcode_source=barcode_source,
                                )
                                outputs = await run_full_pipeline(payload)
                        elif mode == "text_only":
                            payload = build_payload(
                                sample_dir=sample_dir,
                                metadata=metadata,
                                text=text,
                                include_image=False,
                                barcode_source=barcode_source,
                            )
                            outputs = await run_full_pipeline(payload)

                        if outputs is None:
                            raise RuntimeError("No outputs produced.")

                        write_json(
                            raw_path,
                            {
                                "sample_id": sample_id,
                                "mode": mode,
                                "variant": variant_key,
                                "variant_name": variant_name,
                                "metadata": metadata,
                                "outputs": {
                                    key: value.model_dump()
                                    for key, value in outputs.items()
                                },
                            },
                        )
                    except Exception as exc:
                        error = str(exc)
                        failures.append(
                            {
                                "sample_id": sample_id,
                                "mode": mode,
                                "variant": variant_key,
                                "error": error,
                            }
                        )

                    rows.append(
                        case_metrics(
                            sample_id=sample_id,
                            variant_key=variant_key,
                            variant_name=variant_name,
                            mode=mode,
                            barcode_source=barcode_source,
                            barcode_evidence_available=barcode_enabled(metadata, barcode_source),
                            text=text,
                            metadata=metadata,
                            outputs=outputs,
                            error=error,
                        )
                    )
                    persist_results(output_dir, rows, failures)
    finally:
        persist_results(output_dir, rows, failures)

    print(f"\nWrote evaluation results to: {output_dir}")
    return output_dir


def main() -> None:
    args = parse_args()
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
