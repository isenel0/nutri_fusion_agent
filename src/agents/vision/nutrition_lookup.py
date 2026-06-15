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


MANUAL_PRIORS: dict[str, NutritionPrior] = {
    "spinach": NutritionPrior(
        source_name="spinach",
        calories_per_g=0.23,
        fat_per_g=0.004,
        carbs_per_g=0.036,
        protein_per_g=0.029,
    ),
    "green olive": NutritionPrior(
        source_name="green olive",
        calories_per_g=1.4545,
        fat_per_g=0.1527,
        carbs_per_g=0.0382,
        protein_per_g=0.0109,
    ),
    "black olive": NutritionPrior(
        source_name="black olive",
        calories_per_g=1.1514,
        fat_per_g=0.1071,
        carbs_per_g=0.0631,
        protein_per_g=0.008,
    ),
    "kasar cheese": NutritionPrior(
        source_name="kasar cheese",
        calories_per_g=3.5769,
        fat_per_g=0.2692,
        carbs_per_g=0.0192,
        protein_per_g=0.25,
    ),
    "beef doner": NutritionPrior(
        source_name="beef doner",
        calories_per_g=2.5,
        fat_per_g=0.17,
        carbs_per_g=0.05,
        protein_per_g=0.19,
    ),
    "artichoke with olive oil": NutritionPrior(
        source_name="artichoke with olive oil",
        calories_per_g=0.8506,
        fat_per_g=0.0403,
        carbs_per_g=0.1,
        protein_per_g=0.024,
    ),
    "pickled jalapeno pepper": NutritionPrior(
        source_name="pickled jalapeno pepper",
        calories_per_g=0.2706,
        fat_per_g=0.0041,
        carbs_per_g=0.0591,
        protein_per_g=0.0091,
    ),
    "green plum": NutritionPrior(
        source_name="green plum",
        calories_per_g=0.46,
        fat_per_g=0.003,
        carbs_per_g=0.114,
        protein_per_g=0.007,
    ),
    "boiled pasta": NutritionPrior(
        source_name="boiled pasta",
        calories_per_g=1.58,
        fat_per_g=0.009,
        carbs_per_g=0.31,
        protein_per_g=0.058,
    ),
    "rice pilaf": NutritionPrior(
        source_name="rice pilaf",
        calories_per_g=1.3,
        fat_per_g=0.0027,
        carbs_per_g=0.28,
        protein_per_g=0.0273,
    ),
    "boiled chicken breast": NutritionPrior(
        source_name="boiled chicken breast",
        calories_per_g=1.6504,
        fat_per_g=0.0358,
        carbs_per_g=0.0,
        protein_per_g=0.3102,
    ),
    "purslane": NutritionPrior(
        source_name="purslane",
        calories_per_g=0.2,
        fat_per_g=0.0035,
        carbs_per_g=0.0365,
        protein_per_g=0.023,
    ),
    "parsley": NutritionPrior(
        source_name="parsley",
        calories_per_g=0.35,
        fat_per_g=0.0075,
        carbs_per_g=0.0625,
        protein_per_g=0.03,
    ),
    "cooked chickpeas": NutritionPrior(
        source_name="cooked chickpeas",
        calories_per_g=1.6385,
        fat_per_g=0.0262,
        carbs_per_g=0.2738,
        protein_per_g=0.0892,
    ),
    "walnut": NutritionPrior(
        source_name="walnut",
        calories_per_g=6.54,
        fat_per_g=0.652,
        carbs_per_g=0.137,
        protein_per_g=0.152,
    ),
}


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
    "ıspanak": "spinach",
    "ispanak": "spinach",
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
    "yeşil zeytin": "green olive",
    "yesil zeytin": "green olive",
    "siyah zeytin": "black olive",
    "kaşar peyniri": "kasar cheese",
    "kasar peyniri": "kasar cheese",
    "cheddar cheese": "kasar cheese",
    "et döner": "beef doner",
    "döner": "beef doner",
    "doner": "beef doner",
    "lamb doner meat": "beef doner",
    "doner meat": "beef doner",
    "meat doner kebab": "beef doner",
    "zeytinyağlı enginar": "artichoke with olive oil",
    "zeytinyagli enginar": "artichoke with olive oil",
    "artichoke hearts with olive oil": "artichoke with olive oil",
    "jalapeno biber": "pickled jalapeno pepper",
    "yeşil erik": "green plum",
    "yesil erik": "green plum",
    "haşlanmış makarna": "boiled pasta",
    "haslanmis makarna": "boiled pasta",
    "makarna": "boiled pasta",
    "plain pasta cooked": "boiled pasta",
    "pirinç pilavı": "rice pilaf",
    "pirinc pilavi": "rice pilaf",
    "pilav": "rice pilaf",
    "haşlanmış tavuk göğsü": "boiled chicken breast",
    "haslanmis tavuk gogsu": "boiled chicken breast",
    "boiled chicken breast": "boiled chicken breast",
    "chicken breast": "boiled chicken breast",
    "semizotu": "purslane",
    "purslane leaves": "purslane",
    "maydanoz": "parsley",
    "nohut": "cooked chickpeas",
    "chickpeas": "cooked chickpeas",
    "chickpeas cooked": "cooked chickpeas",
    "ceviz": "walnut",
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
        if key in MANUAL_PRIORS:
            return MANUAL_PRIORS[key]

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
