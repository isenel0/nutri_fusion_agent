"""Text agent backed by a local BERT token-classification model."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from schemas import (
    AgentResponse,
    TextAgentData,
    TextEntity,
    TextInputSummary,
    TextMealParse,
    TextModelInfo,
)


class TextAgent(BaseAgent):
    """Extracts food-context entities from free-form meal text."""

    LABEL_TO_TYPE = {
        "FOOD": "food",
        "QUANTITY": "quantity",
        "UNIT": "unit",
        "METHOD": "method",
        "MODIFIER": "modifier",
    }

    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parent
        self.labels = self._load_labels()
        self._pipeline = None

    async def process(self, input_data: Any) -> AgentResponse:
        raw_text = self._normalize_input(input_data)
        if not raw_text:
            contract = self._contract(
                raw_text="",
                entities=[],
                notes=["No text was provided."],
                missing_requirements=["text"],
            )
            return AgentResponse(
                source="text",
                confidence=0.0,
                data=contract.model_dump(),
                error="No text was provided.",
            )

        try:
            entities = await asyncio.to_thread(self._extract_entities, raw_text)
            confidence = self._estimate_confidence(entities)
            contract = self._contract(
                raw_text=raw_text,
                entities=entities,
                notes=[
                    "Entities were extracted by the local token-classification model.",
                    "Use modifiers and methods as context, not as standalone ingredients.",
                ],
            )
            return AgentResponse(
                source="text",
                confidence=confidence,
                data=contract.model_dump(),
                error=None,
            )
        except Exception as exc:
            contract = self._contract(
                raw_text=raw_text,
                entities=[],
                notes=[
                    "Text model inference failed.",
                    "The orchestrator may still use raw_text for LLM-only reasoning if needed.",
                ],
                missing_requirements=["text_model_runtime"],
            )
            return AgentResponse(
                source="text",
                confidence=0.0,
                data=contract.model_dump(),
                error=f"Text analysis failed: {exc}",
            )

    def _normalize_input(self, input_data: Any) -> str:
        if input_data is None:
            return ""
        if isinstance(input_data, str):
            return input_data.strip()
        if isinstance(input_data, dict):
            return str(input_data.get("text") or input_data.get("raw_text") or "").strip()
        return str(input_data).strip()

    def _extract_entities(self, raw_text: str) -> list[TextEntity]:
        extractor = self._load_pipeline()
        predictions = extractor(raw_text)
        entities: list[TextEntity] = []

        for prediction in predictions:
            raw_label = (
                prediction.get("entity_group")
                or prediction.get("entity")
                or ""
            )
            label = raw_label.replace("B-", "").replace("I-", "")
            entity_type = self.LABEL_TO_TYPE.get(label)
            if entity_type is None:
                continue

            start = prediction.get("start")
            end = prediction.get("end")
            if isinstance(start, int) and isinstance(end, int):
                text = raw_text[start:end].strip()
            else:
                text = str(prediction.get("word") or prediction.get("text") or "").strip()
                text = text.replace(" ##", "").replace("##", "")
            if not text:
                continue

            entities.append(
                TextEntity(
                    type=entity_type,
                    text=text,
                    confidence=float(prediction.get("score", 0.0)),
                    start_char=start,
                    end_char=end,
                )
            )

        return self._augment_entities(raw_text, entities)

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        try:
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
                pipeline,
            )
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for the local text model. "
                "Install project requirements before running TextAgent."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
            use_fast=True,
        )
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self._pipeline = pipeline(
            task="token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )
        return self._pipeline

    def _contract(
        self,
        raw_text: str,
        entities: list[TextEntity],
        notes: list[str],
        missing_requirements: list[str] | None = None,
    ) -> TextAgentData:
        return TextAgentData(
            input=TextInputSummary(raw_text=raw_text),
            model=TextModelInfo(
                model_path=str(self.model_dir),
                labels=self.labels,
            ),
            entities=entities,
            parsed=self._parsed_entities(raw_text, entities),
            llm_notes=notes,
            missing_requirements=missing_requirements or [],
        )

    def _parsed_entities(self, raw_text: str, entities: list[TextEntity]) -> TextMealParse:
        grouped: dict[str, list[str]] = {
            "food": [],
            "quantity": [],
            "unit": [],
            "method": [],
            "modifier": [],
        }
        for entity in entities:
            grouped[entity.type].append(entity.text)

        return TextMealParse(
            foods=grouped["food"],
            quantities=grouped["quantity"],
            units=grouped["unit"],
            methods=grouped["method"],
            modifiers=grouped["modifier"],
            portion_multiplier=self._infer_portion_multiplier(raw_text, grouped["quantity"]),
        )

    def _infer_portion_multiplier(self, raw_text: str, quantities: list[str]) -> float | None:
        normalized = f"{raw_text} {' '.join(quantities)}".lower()
        if not normalized:
            return None

        if "half" in normalized or "1/2" in normalized or "yarım" in normalized:
            return 0.5
        if "quarter" in normalized or "1/4" in normalized or "çeyrek" in normalized:
            return 0.25
        if "double" in normalized or "2x" in normalized or "iki porsiyon" in normalized:
            return 2.0
        return None

    def _augment_entities(self, raw_text: str, entities: list[TextEntity]) -> list[TextEntity]:
        augmented = list(entities)
        lowered = raw_text.lower()

        method_keywords = [
            "grilled",
            "fried",
            "boiled",
            "baked",
            "roasted",
            "raw",
            "ızgara",
            "kızarmış",
            "haşlanmış",
        ]
        quantity_keywords = ["half", "quarter", "double", "1/2", "1/4", "yarım", "çeyrek"]
        food_phrases = ["olive oil", "zeytinyağı"]

        self._add_keyword_entities(raw_text, lowered, augmented, "method", method_keywords)
        self._add_keyword_entities(raw_text, lowered, augmented, "quantity", quantity_keywords)
        self._add_keyword_entities(raw_text, lowered, augmented, "food", food_phrases)

        return self._dedupe_entities(augmented)

    def _add_keyword_entities(
        self,
        raw_text: str,
        lowered: str,
        entities: list[TextEntity],
        entity_type: str,
        keywords: list[str],
    ) -> None:
        for keyword in keywords:
            start = lowered.find(keyword)
            if start < 0:
                continue
            end = start + len(keyword)
            if any(
                entity.type == entity_type
                and entity.start_char == start
                and entity.end_char == end
                for entity in entities
            ):
                continue
            entities.append(
                TextEntity(
                    type=entity_type,
                    text=raw_text[start:end],
                    confidence=0.75,
                    start_char=start,
                    end_char=end,
                )
            )

    def _dedupe_entities(self, entities: list[TextEntity]) -> list[TextEntity]:
        seen: set[tuple[str, str, int | None, int | None]] = set()
        deduped: list[TextEntity] = []
        for entity in entities:
            key = (
                entity.type,
                entity.text.lower(),
                entity.start_char,
                entity.end_char,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entity)
        return deduped

    def _estimate_confidence(self, entities: list[TextEntity]) -> float:
        if not entities:
            return 0.0
        return sum(entity.confidence for entity in entities) / len(entities)

    def _load_labels(self) -> list[str]:
        label_map_path = self.model_dir / "label_map.json"
        if not label_map_path.exists():
            return []
        data = json.loads(label_map_path.read_text(encoding="utf-8"))
        return [str(label) for label in data.get("labels", [])]
