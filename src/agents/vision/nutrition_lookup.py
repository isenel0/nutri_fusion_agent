"""Nutrition-prior lookup for FoodSeg-style ingredient labels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NutritionPrior:
    """Per-gram nutrition values used to estimate macros from mass."""

    source_name: str
    calories_per_g: float
    fat_per_g: float
    carbs_per_g: float
    protein_per_g: float
    matched: bool = True


DEFAULT_PRIOR = NutritionPrior(
    source_name="generic mixed food",
    calories_per_g=2.0,
    fat_per_g=0.08,
    carbs_per_g=0.18,
    protein_per_g=0.08,
    matched=False,
)


ALIASES: dict[str, str | tuple[str, ...]] = {
    "candy": "candies",
    "egg tart": "pies",
    "cheese butter": ("cheese", "butter"),
    "juice": "apple juice",
    "red beans": "kidney beans",
    "dried cranberries": "cranberries",
    "soy": "soy sauce",
    "egg": "eggs",
    "strawberry": "strawberries",
    "cherry": "cherries",
    "blueberry": "blueberries",
    "raspberry": "raspberries",
    "fig": "figs",
    "melon": "cantaloupe",
    "chicken duck": ("chicken", "duck"),
    "fried meat": "beef",
    "sauce": "barbecue sauce",
    "shellfish": "shrimp",
    "soup": "broth",
    "hamburg": "hamburger",
    "hanamaki baozi": "dumplings",
    "wonton dumplings": "dumplings",
    "pie": "apple pie",
    "potato": "potatoes",
    "tomato": "tomatoes",
    "kelp": "seaweed",
    "spring onion": "green onions",
    "rape": "bok choy",
    "white radish": "radishes",
    "bamboo shoots": "bamboo shoot",
    "celery stick": "celery",
    "cilantro mint": ("cilantro", "mint"),
    "french beans": "green beans",
    "king oyster mushroom": "mushrooms",
    "shiitake": "mushrooms",
    "enoki mushroom": "mushrooms",
    "oyster mushroom": "mushrooms",
    "white button mushroom": "mushrooms",
    "salad": "caesar salad",
    "other ingredients": "generic mixed food",
}


class NutritionLookup:
    """Maps segmentation labels to Nutrition5K per-gram macro priors."""

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            db_path = Path(__file__).resolve().parents[3] / "nutrition5k" / "metadata" / "nutrition_db.json"
        self.db_path = db_path
        self._entries = self._load_entries(db_path)

    def lookup(self, label: str) -> NutritionPrior:
        key = self._normalize(label)
        alias = ALIASES.get(key, key)

        if isinstance(alias, tuple):
            priors = [self._lookup_one(item) for item in alias]
            matched = [prior for prior in priors if prior.matched]
            if not matched:
                return DEFAULT_PRIOR
            return self._average_priors(label, matched)

        return self._lookup_one(alias)

    def _lookup_one(self, label: str) -> NutritionPrior:
        key = self._normalize(label)
        if key == DEFAULT_PRIOR.source_name:
            return DEFAULT_PRIOR

        entry = self._entries.get(key)
        source_name = key

        if entry is None:
            close = get_close_matches(key, self._entries.keys(), n=1, cutoff=0.88)
            if close:
                source_name = close[0]
                entry = self._entries[source_name]

        if entry is None:
            return DEFAULT_PRIOR

        return NutritionPrior(
            source_name=source_name,
            calories_per_g=float(entry.get("cal_per_g", 0.0)),
            fat_per_g=float(entry.get("fat_per_g", 0.0)),
            carbs_per_g=float(entry.get("carb_per_g", 0.0)),
            protein_per_g=float(entry.get("protein_per_g", 0.0)),
        )

    def _average_priors(self, source_name: str, priors: list[NutritionPrior]) -> NutritionPrior:
        count = len(priors)
        return NutritionPrior(
            source_name=source_name,
            calories_per_g=sum(prior.calories_per_g for prior in priors) / count,
            fat_per_g=sum(prior.fat_per_g for prior in priors) / count,
            carbs_per_g=sum(prior.carbs_per_g for prior in priors) / count,
            protein_per_g=sum(prior.protein_per_g for prior in priors) / count,
            matched=all(prior.matched for prior in priors),
        )

    def _load_entries(self, db_path: Path) -> dict[str, dict[str, Any]]:
        with db_path.open("r", encoding="utf-8") as file:
            raw_entries = json.load(file)
        return {
            self._normalize(name): values
            for name, values in raw_entries.items()
            if isinstance(values, dict)
        }

    def _normalize(self, value: str) -> str:
        return " ".join(value.strip().lower().replace("_", " ").split())
