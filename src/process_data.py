from __future__ import annotations

import argparse
import csv
from pathlib import Path


BASE_FIELDS = [
	"dish_id",
	"total_calories",
	"total_mass",
	"total_fat",
	"total_carb",
	"total_protein",
]

INGREDIENT_FIELDS = [
	"id",
	"name",
	"grams",
	"calories",
	"fat",
	"carb",
	"protein",
]


def has_header(row: list[str]) -> bool:
	"""Return True when the first row appears to be a header row."""
	if not row:
		return False
	first = row[0].strip().lower()
	return first in {"dish_id", "ingr", "ingredient"}


def infer_header_layout(data_row: list[str]) -> tuple[bool, int]:
	"""
	Infer whether a `num_ingrs` column exists and how many ingredient blocks exist.

	Expected base fields:
	  - without count: 6 fields
	  - with count: 7 fields (`num_ingrs` as 7th field)
	Each ingredient contributes 7 repeated fields.
	"""
	total_cols = len(data_row)
	if total_cols < 6:
		raise ValueError(f"Row has too few columns ({total_cols}).")

	no_count_remainder = total_cols - len(BASE_FIELDS)
	with_count_remainder = total_cols - (len(BASE_FIELDS) + 1)

	no_count_valid = no_count_remainder >= 0 and no_count_remainder % len(INGREDIENT_FIELDS) == 0
	with_count_valid = with_count_remainder >= 0 and with_count_remainder % len(INGREDIENT_FIELDS) == 0

	if with_count_valid and not no_count_valid:
		return True, with_count_remainder // len(INGREDIENT_FIELDS)
	if no_count_valid and not with_count_valid:
		return False, no_count_remainder // len(INGREDIENT_FIELDS)

	if with_count_valid and no_count_valid:
		# Ambiguous case; prefer explicit count when the candidate looks numeric.
		count_candidate = data_row[len(BASE_FIELDS)].strip()
		if count_candidate.isdigit():
			return True, with_count_remainder // len(INGREDIENT_FIELDS)
		return False, no_count_remainder // len(INGREDIENT_FIELDS)

	raise ValueError(
		"Could not infer schema; column count does not match expected Nutrition5k layout."
	)


def build_header(has_num_ingrs: bool, ingredient_count: int) -> list[str]:
	header = list(BASE_FIELDS)
	if has_num_ingrs:
		header.append("num_ingrs")

	for i in range(1, ingredient_count + 1):
		for field in INGREDIENT_FIELDS:
			header.append(f"ingr_{i}_{field}")
	return header


def process_file(input_path: Path, output_path: Path, overwrite: bool) -> None:
	with input_path.open("r", encoding="utf-8", newline="") as f:
		rows = list(csv.reader(f))

	if not rows:
		print(f"Skipped empty file: {input_path}")
		return

	first_row = rows[0]
	if has_header(first_row):
		if overwrite:
			print(f"Already has header, leaving as-is: {input_path}")
		else:
			with output_path.open("w", encoding="utf-8", newline="") as out:
				writer = csv.writer(out)
				writer.writerows(rows)
			print(f"Header already present, copied unchanged: {output_path}")
		return

	has_num_ingrs, ingredient_count = infer_header_layout(first_row)
	header = build_header(has_num_ingrs, ingredient_count)

	if overwrite:
		write_path = input_path
	else:
		write_path = output_path

	with write_path.open("w", encoding="utf-8", newline="") as out:
		writer = csv.writer(out)
		writer.writerow(header)
		writer.writerows(rows)

	print(
		f"Wrote header to {write_path} "
		f"(num_ingrs={'yes' if has_num_ingrs else 'no'}, ingredients={ingredient_count})"
	)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Add inferred field names to Nutrition5k dish metadata CSV files."
	)
	parser.add_argument(
		"--metadata-dir",
		type=Path,
		default=Path("nutri-fusion-agent/nutrition5k/metadata"),
		help="Directory containing dish_metadata_*.csv files.",
	)
	parser.add_argument(
		"--pattern",
		default="dish_metadata_*.csv",
		help="Glob pattern for metadata files to process.",
	)
	parser.add_argument(
		"--suffix",
		default="_with_headers",
		help="Suffix for output files when not overwriting.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite original CSV files instead of writing new files.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if not args.metadata_dir.exists():
		raise FileNotFoundError(f"Metadata directory does not exist: {args.metadata_dir}")

	files = sorted(args.metadata_dir.glob(args.pattern))
	if not files:
		print(f"No files found in {args.metadata_dir} matching pattern: {args.pattern}")
		return

	for file_path in files:
		output_path = file_path.with_name(f"{file_path.stem}{args.suffix}{file_path.suffix}")
		process_file(file_path, output_path, overwrite=args.overwrite)


if __name__ == "__main__":
	main()
