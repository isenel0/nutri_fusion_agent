"""Barcode scanning utilities for image-based decoding."""

from __future__ import annotations

import io
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import cv2
    import numpy as np

    HAS_OPENCV = True
except Exception:
    cv2 = None
    np = None
    HAS_OPENCV = False


class BarcodeScanner:
    """Extract barcode values from an image path or in-memory bytes."""

    NUMERIC_BARCODE_PATTERN = re.compile(r"^\d{6,14}$")
    ALPHANUMERIC_BARCODE_PATTERN = re.compile(r"^[A-Za-z0-9]{3,30}$")
    ROTATION_ANGLES = [90, 180, 270, -15, 15, -30, 30]

    def scan(self, image_path: str | None = None, image_bytes: bytes | None = None) -> str:
        if not image_path and image_bytes is None:
            raise ValueError("Provide either image_path or image_bytes for barcode scanning.")

        with self._resolve_image(image_path=image_path, image_bytes=image_bytes) as image:
            variants = self._build_variants(image)
            opencv_variants = self._build_opencv_variants(image)

        all_variants = variants + opencv_variants
        try:
            from pyzbar.pyzbar import decode

            result = self._scan_with_pyzbar(variants=all_variants, decode=decode)
            if result:
                return result
        except ImportError:
            pass

        return self._scan_with_zbarimg(variants=all_variants)

    def _resolve_image(
        self, image_path: str | None, image_bytes: bytes | None
    ) -> Image.Image:
        if image_bytes is not None:
            return Image.open(io.BytesIO(image_bytes))

        path = Path(image_path or "")
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        return Image.open(path)

    def _build_variants(self, image: Image.Image) -> list[tuple[str, Image.Image]]:
        base = image.convert("RGB")
        gray = ImageOps.grayscale(base)
        gray_auto = ImageOps.autocontrast(gray)

        variants: list[tuple[str, Image.Image]] = [
            ("gray", gray),
            ("gray_auto", gray_auto),
            ("gray_sharp", gray_auto.filter(ImageFilter.SHARPEN)),
            (
                "gray_unsharp",
                gray_auto.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=3)),
            ),
            ("gray_median", gray_auto.filter(ImageFilter.MedianFilter(size=3))),
            ("contrast_15", ImageEnhance.Contrast(gray_auto).enhance(1.5)),
            ("contrast_20", ImageEnhance.Contrast(gray_auto).enhance(2.0)),
            ("contrast_25", ImageEnhance.Contrast(gray_auto).enhance(2.5)),
            ("inverted", ImageOps.invert(gray_auto)),
        ]

        # Add a few binary variants across thresholds because difficult images can
        # fail with a single threshold setting.
        threshold_source = ImageEnhance.Contrast(gray_auto).enhance(2.0)
        for threshold in [90, 110, 130, 150, 170]:
            bw = self._threshold(threshold_source, threshold=threshold)
            variants.append((f"bw_{threshold}", bw))
            variants.append((f"bw_inv_{threshold}", ImageOps.invert(bw)))

        # Barcode may occupy only part of the frame; crop candidate bands and center.
        crops = self._build_crops(gray_auto)
        for idx, crop in enumerate(crops):
            variants.append((f"crop_{idx}", crop))
            variants.append((f"crop_{idx}_sharp", crop.filter(ImageFilter.SHARPEN)))

        # Upscale helps when bars are small in the original image.
        upscale_2x = gray_auto.resize(
            (max(1, gray_auto.width * 2), max(1, gray_auto.height * 2)),
            Image.Resampling.LANCZOS,
        )
        variants.append(("upscale2x", upscale_2x))
        variants.append(("upscale2x_sharp", upscale_2x.filter(ImageFilter.SHARPEN)))

        return variants[:120]

    def _adaptive_threshold(self, image: Image.Image) -> Image.Image:
        pixels = list(image.getdata())
        avg = sum(pixels) / len(pixels)
        threshold = int(avg * 0.7)
        threshold = max(100, min(180, threshold))
        return image.point(lambda p: 255 if p > threshold else 0)

    def _threshold(self, image: Image.Image, threshold: int) -> Image.Image:
        return image.point(lambda p: 255 if p > threshold else 0)

    def _build_crops(self, image: Image.Image) -> list[Image.Image]:
        width, height = image.size
        if width < 20 or height < 20:
            return [image]

        boxes = [
            # Center region.
            (int(width * 0.15), int(height * 0.20), int(width * 0.85), int(height * 0.80)),
            # Horizontal bands often capture barcodes on product labels.
            (0, int(height * 0.15), width, int(height * 0.55)),
            (0, int(height * 0.30), width, int(height * 0.70)),
            (0, int(height * 0.45), width, int(height * 0.85)),
            # Vertical center strip.
            (int(width * 0.25), 0, int(width * 0.75), height),
        ]

        crops: list[Image.Image] = []
        for left, top, right, bottom in boxes:
            if right - left < 20 or bottom - top < 20:
                continue
            crops.append(image.crop((left, top, right, bottom)))

        return crops

    def _build_opencv_variants(self, image: Image.Image) -> list[tuple[str, Image.Image]]:
        """Build barcode-focused ROI variants via OpenCV morphology and contours.

        This fallback is especially useful for difficult shots where the barcode
        occupies a smaller region or has local contrast issues.
        """
        if not HAS_OPENCV:
            return []

        rgb = image.convert("RGB")
        frame = np.array(rgb)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=-1)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=-1)
        gradient = cv2.subtract(grad_x, grad_y)
        gradient = cv2.convertScaleAbs(gradient)

        blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
        _, thresh = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morphed = cv2.erode(morphed, None, iterations=2)
        morphed = cv2.dilate(morphed, None, iterations=2)

        contours, _ = cv2.findContours(
            morphed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        area_limit = gray.shape[0] * gray.shape[1]
        top_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        variants: list[tuple[str, Image.Image]] = []
        roi_index = 0
        for contour in top_contours:
            if cv2.contourArea(contour) < area_limit * 0.01:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < 40 or h < 20:
                continue

            # Axis-aligned ROI.
            roi = gray[max(0, y - 4) : min(gray.shape[0], y + h + 4), max(0, x - 4) : min(gray.shape[1], x + w + 4)]
            if roi.size > 0:
                pil_roi = Image.fromarray(roi)
                variants.append((f"cv_roi_{roi_index}", ImageOps.autocontrast(pil_roi)))
                variants.append(
                    (
                        f"cv_roi_{roi_index}_sharp",
                        ImageOps.autocontrast(pil_roi).filter(ImageFilter.SHARPEN),
                    )
                )
                variants.append(
                    (
                        f"cv_roi_{roi_index}_bin",
                        self._threshold(ImageOps.autocontrast(pil_roi), 130),
                    )
                )
                roi_index += 1

            # Rotated ROI based on min-area rectangle.
            rect = cv2.minAreaRect(contour)
            warped = self._extract_rotated_rect(gray, rect)
            if warped is not None and warped.size > 0:
                pil_warped = Image.fromarray(warped)
                variants.append((f"cv_warp_{roi_index}", ImageOps.autocontrast(pil_warped)))
                variants.append(
                    (
                        f"cv_warp_{roi_index}_sharp",
                        ImageOps.autocontrast(pil_warped).filter(ImageFilter.SHARPEN),
                    )
                )
                roi_index += 1

        return variants[:40]

    def _extract_rotated_rect(self, gray_image, rect):
        if not HAS_OPENCV:
            return None

        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="float32")

        width = int(max(rect[1][0], rect[1][1]))
        height = int(min(rect[1][0], rect[1][1]))
        if width < 30 or height < 12:
            return None

        src = self._order_points(box)
        dst = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype="float32",
        )

        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(gray_image, matrix, (width, height))

        # Ensure horizontal orientation for 1D barcodes.
        if warped.shape[0] > warped.shape[1]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        return warped

    def _order_points(self, points):
        if not HAS_OPENCV:
            return points

        rect = np.zeros((4, 2), dtype="float32")
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]

        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]
        return rect

    def _is_valid_barcode(self, value: str) -> bool:
        return bool(
            self.NUMERIC_BARCODE_PATTERN.match(value)
            or self.ALPHANUMERIC_BARCODE_PATTERN.match(value)
        )

    def _add_rotations(
        self, variants: list[tuple[str, Image.Image]]
    ) -> list[tuple[str, Image.Image]]:
        rotated: list[tuple[str, Image.Image]] = []
        for name, image in variants:
            for angle in self.ROTATION_ANGLES:
                rotated.append((f"{name}_rot{angle}", image.rotate(angle, expand=True)))
        return variants + rotated

    def _scan_with_pyzbar(
        self,
        variants: list[tuple[str, Image.Image]],
        decode,
    ) -> str | None:
        for _, image in variants:
            decoded = self._try_decode(image, decode)
            if decoded:
                return decoded

        for _, image in self._add_rotations(variants):
            decoded = self._try_decode(image, decode)
            if decoded:
                return decoded

        return None

    def _try_decode(self, image: Image.Image, decode) -> str | None:
        try:
            detections = decode(image)
        except Exception:
            return None

        for detection in detections:
            try:
                value = detection.data.decode("utf-8").strip()
            except UnicodeDecodeError:
                continue

            if value and self._is_valid_barcode(value):
                return value

        return None

    def _scan_with_zbarimg(self, variants: list[tuple[str, Image.Image]]) -> str:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            for idx, (_, image) in enumerate(variants):
                candidate = temp_dir_path / f"candidate_{idx}.png"
                image.save(candidate, format="PNG")
                value = self._run_zbarimg(candidate)
                if value and self._is_valid_barcode(value):
                    return value

            for idx, (_, image) in enumerate(self._add_rotations(variants)):
                candidate = temp_dir_path / f"rotated_{idx}.png"
                image.save(candidate, format="PNG")
                value = self._run_zbarimg(candidate)
                if value and self._is_valid_barcode(value):
                    return value

        raise ValueError(
            "No barcode detected after preprocessing attempts. "
            "Try a closer, sharper image with better lighting."
        )

    def _run_zbarimg(self, image_path: Path) -> str | None:
        try:
            proc = subprocess.run(
                ["zbarimg", "--quiet", "--raw", str(image_path)],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Barcode decoding backend unavailable. Install zbar (e.g. brew install zbar) "
                "and ensure zbarimg is in PATH."
            ) from exc

        output = (proc.stdout or "").strip()
        if proc.returncode != 0 or not output:
            return None

        first_line = output.splitlines()[0].strip()
        return first_line or None
