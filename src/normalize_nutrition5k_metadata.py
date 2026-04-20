from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


BASE_FIELDS = [
    "dish_id",
    "total_calories",
    "total_mass",
    "total_fat",
    "total_carb",
    "total_protein",
]

ING_BLOCK_SIZE = 7

ING_FIELDS = [
    "ingr_id_raw",
    "ingr_name_raw",
    "grams",
    "calories",
    "fat",
    "carb",
    "protein",
]

INGR_ID_RE = re.compile(r"(\d+)$")


@dataclass(frozen=True)
class IngredientMeta:
    ingredient_id: int
    name: str
    cal_per_g: float
    fat_per_g: float
    carb_per_g: float
    protein_per_g: float


def parse_float(value: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_ingredient_id(raw: str) -> Optional[int]:
    raw = raw.strip()
    if not raw:
        return None
    # Expected: ingr_0000000122 -> 122
    m = INGR_ID_RE.search(raw)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def normalize_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def load_ingredients_metadata(path: Path) -> dict[int, IngredientMeta]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"ingr", "id", "cal/g", "fat(g)", "carb(g)", "protein(g)"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Unexpected ingredients_metadata schema in {path}: {reader.fieldnames}"
            )

        out: dict[int, IngredientMeta] = {}
        for row in reader:
            ingredient_id = int(row["id"])
            out[ingredient_id] = IngredientMeta(
                ingredient_id=ingredient_id,
                name=normalize_whitespace(row["ingr"]).lower(),
                cal_per_g=float(row["cal/g"]),
                fat_per_g=float(row["fat(g)"]),
                carb_per_g=float(row["carb(g)"]),
                protein_per_g=float(row["protein(g)"]),
            )
    return out


def iter_dish_rows(path: Path) -> Iterable[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader, start=1):
            if not row:
                continue
            # Skip header if present
            if i == 1 and row[0].strip().lower() == "dish_id":
                continue
            yield row


def parse_dish_row(row: list[str], *, source: str) -> tuple[dict, list[dict]]:
    if len(row) < len(BASE_FIELDS):
        raise ValueError(f"Row too short ({len(row)}) in {source}: {row[:10]}")

    dish_id = row[0].strip()
    base_raw = row[: len(BASE_FIELDS)]
    rest = row[len(BASE_FIELDS) :]

    if len(rest) % ING_BLOCK_SIZE != 0:
        raise ValueError(
            f"Ingredient remainder not multiple of {ING_BLOCK_SIZE} in {source} dish {dish_id}: "
            f"rest_cols={len(rest)} total_cols={len(row)}"
        )

    dish = {
        "dish_id": dish_id,
        "orig_total_calories": parse_float(base_raw[1]),
        "orig_total_mass": parse_float(base_raw[2]),
        "orig_total_fat": parse_float(base_raw[3]),
        "orig_total_carb": parse_float(base_raw[4]),
        "orig_total_protein": parse_float(base_raw[5]),
        "source": source,
    }

    ingredients: list[dict] = []
    num_blocks = len(rest) // ING_BLOCK_SIZE
    for block_index in range(num_blocks):
        start = block_index * ING_BLOCK_SIZE
        block = rest[start : start + ING_BLOCK_SIZE]
        ing = {
            "dish_id": dish_id,
            "ingr_index": block_index + 1,
            "ingr_id_raw": block[0].strip(),
            "ingr_name_raw": normalize_whitespace(block[1]).lower(),
            "grams": parse_float(block[2]) or 0.0,
            "calories": parse_float(block[3]) or 0.0,
            "fat": parse_float(block[4]) or 0.0,
            "carb": parse_float(block[5]) or 0.0,
            "protein": parse_float(block[6]) or 0.0,
            "source": source,
        }
        ingredients.append(ing)

    dish["num_ingredients"] = len(ingredients)
    return dish, ingredients


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def build_outputs(
    dish_paths: list[Path],
    ingredients_meta: dict[int, IngredientMeta],
) -> tuple[list[dict], list[dict]]:
    dishes_out: list[dict] = []
    ing_out: list[dict] = []

    for dish_path in dish_paths:
        source = dish_path.stem
        for row in iter_dish_rows(dish_path):
            dish, ingredients = parse_dish_row(row, source=source)

            sum_mass = sum(i["grams"] for i in ingredients)
            sum_cal = sum(i["calories"] for i in ingredients)
            sum_fat = sum(i["fat"] for i in ingredients)
            sum_carb = sum(i["carb"] for i in ingredients)
            sum_protein = sum(i["protein"] for i in ingredients)

            dish["sum_mass"] = sum_mass
            dish["sum_calories"] = sum_cal
            dish["sum_fat"] = sum_fat
            dish["sum_carb"] = sum_carb
            dish["sum_protein"] = sum_protein

            # For training, use totals computed from ingredient rows (consistent across cafes
            # and robust to occasional corrupt totals in the source files).
            dish["total_mass"] = sum_mass
            dish["total_calories"] = sum_cal
            dish["total_fat"] = sum_fat
            dish["total_carb"] = sum_carb
            dish["total_protein"] = sum_protein

            # Diagnostics: how far the provided dish-level totals disagree with the summed
            # ingredient values. (orig - sum)
            dish["delta_calories"] = (dish["orig_total_calories"] or 0.0) - sum_cal
            dish["delta_mass"] = (dish["orig_total_mass"] or 0.0) - sum_mass
            dish["delta_fat"] = (dish["orig_total_fat"] or 0.0) - sum_fat
            dish["delta_carb"] = (dish["orig_total_carb"] or 0.0) - sum_carb
            dish["delta_protein"] = (dish["orig_total_protein"] or 0.0) - sum_protein

            dishes_out.append(dish)

            for ing in ingredients:
                ingredient_id = parse_ingredient_id(ing["ingr_id_raw"])
                meta = ingredients_meta.get(ingredient_id) if ingredient_id else None

                ing_row = dict(ing)
                ing_row["ingredient_id"] = ingredient_id
                ing_row["ingredient_name"] = meta.name if meta else ""
                ing_row["is_unknown_ingredient"] = 0 if meta else 1
                ing_row["is_deprecated"] = 1 if ing["ingr_name_raw"] == "deprecated" else 0

                # Computed nutrition from canonical metadata (optional feature)
                if meta:
                    g = float(ing_row["grams"])
                    ing_row["meta_calories"] = meta.cal_per_g * g
                    ing_row["meta_fat"] = meta.fat_per_g * g
                    ing_row["meta_carb"] = meta.carb_per_g * g
                    ing_row["meta_protein"] = meta.protein_per_g * g
                else:
                    ing_row["meta_calories"] = ""
                    ing_row["meta_fat"] = ""
                    ing_row["meta_carb"] = ""
                    ing_row["meta_protein"] = ""

                ing_out.append(ing_row)

    return dishes_out, ing_out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Normalize Nutrition5k metadata by converting variable-length dish rows into clean CSVs. "
            "Outputs a dish-level table and a long-format dish-ingredient table."
        )
    )
    p.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("nutri-fusion-agent/nutrition5k/metadata"),
        help="Nutrition5k metadata directory.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("nutri-fusion-agent/nutrition5k/metadata/normalized"),
        help="Output directory for normalized CSVs.",
    )
    p.add_argument(
        "--dish-files",
        nargs="+",
        default=["dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv"],
        help="Dish metadata files (relative to metadata-dir).",
    )
    p.add_argument(
        "--ingredients-metadata",
        default="ingredients_metadata.csv",
        help="Ingredients metadata filename (relative to metadata-dir).",
    )
    p.add_argument(
        "--skip-training-files",
        action="store_true",
        help="Skip writing compact training-focused CSVs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    metadata_dir: Path = args.metadata_dir
    out_dir: Path = args.out_dir

    ingredients_path = metadata_dir / args.ingredients_metadata
    ingredients_meta = load_ingredients_metadata(ingredients_path)

    dish_paths = [metadata_dir / name for name in args.dish_files]
    for p in dish_paths:
        if not p.exists():
            raise FileNotFoundError(f"Dish metadata file not found: {p}")

    dishes, ingredients = build_outputs(dish_paths, ingredients_meta)

    dish_out_path = out_dir / "dishes.csv"
    ing_out_path = out_dir / "dish_ingredients_long.csv"

    dish_fields = [
        "dish_id",
        "source",
        "num_ingredients",
        "orig_total_calories",
        "orig_total_mass",
        "orig_total_fat",
        "orig_total_carb",
        "orig_total_protein",
        "sum_calories",
        "sum_mass",
        "sum_fat",
        "sum_carb",
        "sum_protein",
        "total_calories",
        "total_mass",
        "total_fat",
        "total_carb",
        "total_protein",
        "delta_calories",
        "delta_mass",
        "delta_fat",
        "delta_carb",
        "delta_protein",
    ]

    ing_fields = [
        "dish_id",
        "source",
        "ingr_index",
        "ingr_id_raw",
        "ingredient_id",
        "ingr_name_raw",
        "ingredient_name",
        "is_unknown_ingredient",
        "is_deprecated",
        "grams",
        "calories",
        "fat",
        "carb",
        "protein",
        "meta_calories",
        "meta_fat",
        "meta_carb",
        "meta_protein",
    ]

    write_csv(dish_out_path, dish_fields, dishes)
    write_csv(ing_out_path, ing_fields, ingredients)

    print(f"Wrote {len(dishes)} dishes -> {dish_out_path}")
    print(f"Wrote {len(ingredients)} dish-ingredient rows -> {ing_out_path}")

    if not args.skip_training_files:
        training_dish_fields = [
            "dish_id",
            "source",
            "num_ingredients",
            "total_calories",
            "total_mass",
            "total_fat",
            "total_carb",
            "total_protein",
        ]

        training_ing_fields = [
            "dish_id",
            "ingr_index",
            "ingredient_id",
            "ingredient_name",
            "grams",
            "calories",
            "fat",
            "carb",
            "protein",
            "is_deprecated",
        ]

        dishes_training = [
            {key: row.get(key, "") for key in training_dish_fields} for row in dishes
        ]
        ingredients_training = [
            {key: row.get(key, "") for key in training_ing_fields}
            for row in ingredients
            if int(row.get("is_deprecated", 0)) == 0
        ]

        dish_training_out_path = out_dir / "dishes_training.csv"
        ing_training_out_path = out_dir / "dish_ingredients_training.csv"

        write_csv(dish_training_out_path, training_dish_fields, dishes_training)
        write_csv(ing_training_out_path, training_ing_fields, ingredients_training)

        print(
            f"Wrote {len(dishes_training)} training dishes -> {dish_training_out_path}"
        )
        print(
            f"Wrote {len(ingredients_training)} training dish-ingredient rows -> "
            f"{ing_training_out_path}"
        )


if __name__ == "__main__":
    main()
