from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

from .training import TrainingConfig, build_mass_index, mask_path_for_dish


def _fallback_shape(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as img:
        width, height = img.size
    return height, width


def _build_food_mask(result, fallback_shape: tuple[int, int]) -> np.ndarray:
    if result.masks is None or result.masks.data is None:
        return np.zeros(fallback_shape, dtype=np.uint8)

    masks = result.masks.data.cpu().numpy()
    if masks.size == 0:
        return np.zeros(fallback_shape, dtype=np.uint8)

    merged = np.any(masks > 0.5, axis=0)
    return (merged.astype(np.uint8) * 255)


def _save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and cache food masks for Nutrition5k overhead images."
    )
    parser.add_argument(
        "--model-path",
        default="src/agents/vision/best.pt",
        help="Path to the YOLOv8 segmentation model weights.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite masks that already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of samples for a quick run (0 means no limit).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index for a partial run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = TrainingConfig()
    index = build_mass_index(config)

    start = max(args.start, 0)
    samples = index[start:]
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    model = YOLO(args.model_path)

    processed = 0
    skipped = 0

    for sample in samples:
        out_path = mask_path_for_dish(config, sample.dish_id)
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        results = model.predict(str(sample.image_path), conf=0.25, verbose=False)
        result = results[0]

        if result.orig_img is not None:
            height, width = result.orig_img.shape[:2]
        else:
            height, width = _fallback_shape(sample.image_path)

        mask = _build_food_mask(result, (height, width))
        _save_mask(mask, out_path)
        processed += 1

    print("Mask cache completed.")
    print(f"Processed: {processed}")
    print(f"Skipped existing: {skipped}")


if __name__ == "__main__":
    main()
