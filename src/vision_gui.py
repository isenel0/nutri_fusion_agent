"""Small desktop GUI for testing the late-fusion vision agent."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any
import tkinter as tk

from PIL import Image, ImageTk

from agents.vision.agent import VisionAgent
from schemas import AgentResponse


IMAGE_TYPES = [
    ("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
    ("All Files", "*.*"),
]


async def run_vision_agent(
    image_path: str,
    depth_path: str | None,
    confidence: float,
) -> AgentResponse:
    """Run the API-facing VisionAgent with an RGB image and optional depth image."""

    image = Path(image_path)
    if not image.exists() or not image.is_file():
        raise FileNotFoundError(f"RGB image path is invalid: {image_path}")

    payload: dict[str, Any] = {"image_path": image}

    if depth_path:
        depth = Path(depth_path)
        if not depth.exists() or not depth.is_file():
            raise FileNotFoundError(f"Depth image path is invalid: {depth_path}")
        payload["depth_path"] = depth

    agent = VisionAgent(confidence=confidence)
    return await agent.process(payload)


class VisionAgentGUI:
    """Focused Tkinter UI for manually testing vision outputs on local images."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Vision Agent Test GUI")
        self.root.geometry("1120x760")
        self.root.minsize(960, 620)

        self.image_path_var = tk.StringVar(value="")
        self.depth_path_var = tk.StringVar(value="")
        self.confidence_var = tk.DoubleVar(value=0.25)
        self.status_var = tk.StringVar(value="Ready")

        self.rgb_preview: ttk.Label
        self.depth_preview: ttk.Label
        self.summary_output: ScrolledText
        self.json_output: ScrolledText
        self.run_button: ttk.Button
        self._rgb_photo: ImageTk.PhotoImage | None = None
        self._depth_photo: ImageTk.PhotoImage | None = None

        self._build_layout()

    def _build_layout(self) -> None:
        container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = ttk.Frame(container, padding=10)
        right = ttk.Frame(container, padding=10)
        container.add(left, weight=1)
        container.add(right, weight=2)

        ttk.Label(left, text="Vision Inputs", font=("Helvetica", 14, "bold")).pack(
            anchor="w", pady=(0, 10)
        )

        self._build_file_row(
            parent=left,
            label="RGB image",
            variable=self.image_path_var,
            command=self.choose_image,
        )
        self._build_file_row(
            parent=left,
            label="Depth image",
            variable=self.depth_path_var,
            command=self.choose_depth,
        )

        conf_frame = ttk.Frame(left)
        conf_frame.pack(fill=tk.X, pady=(4, 12))
        ttk.Label(conf_frame, text="YOLO confidence").pack(anchor="w")
        conf_control = ttk.Frame(conf_frame)
        conf_control.pack(fill=tk.X, pady=(4, 0))
        ttk.Scale(
            conf_control,
            from_=0.05,
            to=0.95,
            variable=self.confidence_var,
            command=lambda _value: self._update_confidence_label(),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.confidence_label = ttk.Label(conf_control, text="0.25", width=6)
        self.confidence_label.pack(side=tk.LEFT, padx=(8, 0))

        button_row = ttk.Frame(left)
        button_row.pack(fill=tk.X, pady=(0, 10))
        self.run_button = ttk.Button(button_row, text="Run Vision Agent", command=self.run)
        self.run_button.pack(side=tk.LEFT)
        ttk.Button(button_row, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=8)

        ttk.Label(left, textvariable=self.status_var, foreground="#0a4b78").pack(
            anchor="w", pady=(0, 12)
        )

        preview_frame = ttk.Frame(left)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(preview_frame, text="RGB Preview").pack(anchor="w")
        self.rgb_preview = ttk.Label(preview_frame, text="No RGB image selected", anchor="center")
        self.rgb_preview.pack(fill=tk.BOTH, expand=True, pady=(4, 12))

        ttk.Label(preview_frame, text="Depth Preview").pack(anchor="w")
        self.depth_preview = ttk.Label(preview_frame, text="No depth image selected", anchor="center")
        self.depth_preview.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        ttk.Label(right, text="Vision Outputs", font=("Helvetica", 14, "bold")).pack(
            anchor="w", pady=(0, 8)
        )

        tabs = ttk.Notebook(right)
        tabs.pack(fill=tk.BOTH, expand=True)

        summary_tab = ttk.Frame(tabs)
        json_tab = ttk.Frame(tabs)
        tabs.add(summary_tab, text="Summary")
        tabs.add(json_tab, text="Raw JSON")

        self.summary_output = self._build_output_box(summary_tab)
        self.json_output = self._build_output_box(json_tab)
        self._set_text(
            self.summary_output,
            "Select an RGB image and optional depth image, then run the vision agent.",
        )
        self._set_text(self.json_output, "{}")

    def _build_file_row(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        command: Any,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(row, text=label, width=12).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Button(row, text="Browse", command=command).pack(side=tk.LEFT)

    def _build_output_box(self, parent: ttk.Frame) -> ScrolledText:
        box = ScrolledText(parent, wrap=tk.WORD, font=("Menlo", 11), height=20)
        box.pack(fill=tk.BOTH, expand=True)
        box.config(state=tk.DISABLED)
        return box

    def choose_image(self) -> None:
        file_path = filedialog.askopenfilename(title="Select RGB food image", filetypes=IMAGE_TYPES)
        if file_path:
            self.image_path_var.set(file_path)
            self._load_preview(file_path, target="rgb")

    def choose_depth(self) -> None:
        file_path = filedialog.askopenfilename(title="Select depth image", filetypes=IMAGE_TYPES)
        if file_path:
            self.depth_path_var.set(file_path)
            self._load_preview(file_path, target="depth")

    def clear(self) -> None:
        self.image_path_var.set("")
        self.depth_path_var.set("")
        self.status_var.set("Ready")
        self._rgb_photo = None
        self._depth_photo = None
        self.rgb_preview.config(image="", text="No RGB image selected")
        self.depth_preview.config(image="", text="No depth image selected")
        self._set_text(
            self.summary_output,
            "Select an RGB image and optional depth image, then run the vision agent.",
        )
        self._set_text(self.json_output, "{}")

    def run(self) -> None:
        image_path = self.image_path_var.get().strip()
        depth_path = self.depth_path_var.get().strip() or None
        confidence = round(float(self.confidence_var.get()), 2)

        if not image_path:
            messagebox.showwarning("Missing image", "Select an RGB image first.")
            return

        self.run_button.config(state=tk.DISABLED)
        mode = "late fusion" if depth_path else "segmentation only"
        self.status_var.set(f"Running {mode} inference...")

        thread = threading.Thread(
            target=self._run_worker,
            args=(image_path, depth_path, confidence),
            daemon=True,
        )
        thread.start()

    def _run_worker(self, image_path: str, depth_path: str | None, confidence: float) -> None:
        try:
            response = asyncio.run(run_vision_agent(image_path, depth_path, confidence))
        except Exception as exc:
            self.root.after(0, lambda: self._show_failure(exc))
            return

        self.root.after(0, lambda: self._render_response(response))

    def _show_failure(self, exc: Exception) -> None:
        self.run_button.config(state=tk.NORMAL)
        self.status_var.set(f"Failed: {exc}")

    def _render_response(self, response: AgentResponse) -> None:
        payload = response.model_dump()
        self._set_text(self.summary_output, self._render_summary(payload))
        self._set_text(self.json_output, json.dumps(payload, indent=2))
        self.status_var.set("Done")
        self.run_button.config(state=tk.NORMAL)

    def _render_summary(self, payload: dict[str, Any]) -> str:
        data = payload.get("data") or {}
        mode = data.get("mode") or "-"
        totals = data.get("totals") or {}
        detected_items = data.get("detected_items") or []
        ingredients = data.get("ingredients") or []
        ratio_summary = data.get("ratio_summary") or {}
        notes = data.get("llm_notes") or []
        missing = data.get("missing_requirements") or []
        models = data.get("models") or {}

        lines = [
            f"Source: {payload.get('source', 'vision')}",
            f"Mode: {mode}",
            f"Confidence: {self._fmt_number(payload.get('confidence'))}",
            f"Error: {payload.get('error') or '-'}",
            "",
            "Models",
            f"- Segmentation: {models.get('segmentation_model') or '-'}",
            f"- Nutrition: {models.get('nutrition_model') or '-'}",
            f"- Fusion strategy: {models.get('fusion_strategy') or '-'}",
            "",
            "Detected Items",
        ]

        if detected_items:
            for item in detected_items:
                lines.append(
                    f"- {item.get('name', '-')}: "
                    f"{self._fmt_percent(item.get('pixel_ratio'))} pixel ratio"
                )
        else:
            lines.append("- None")

        lines.extend(["", "Totals"])
        if totals:
            lines.extend(
                [
                    f"- Calories (kcal): {self._fmt_number(totals.get('calories_kcal'))}",
                    f"- Mass (g): {self._fmt_number(totals.get('mass_g'))}",
                    f"- Protein (g): {self._fmt_number(totals.get('protein_g'))}",
                    f"- Carbs (g): {self._fmt_number(totals.get('carbs_g'))}",
                    f"- Fat (g): {self._fmt_number(totals.get('fat_g'))}",
                ]
            )
        else:
            lines.append("- Not available without depth image")

        lines.extend(["", "Ingredient Allocation"])
        if ingredients:
            for ingredient in ingredients:
                nutrition = ingredient.get("nutrition") or {}
                lines.extend(
                    [
                        f"- {ingredient.get('name', '-')}",
                        f"  ratio: {self._fmt_percent(ingredient.get('normalized_ratio'))}",
                        f"  kcal: {self._fmt_number(nutrition.get('calories_kcal'))}",
                        f"  mass_g: {self._fmt_number(nutrition.get('mass_g'))}",
                        f"  protein_g: {self._fmt_number(nutrition.get('protein_g'))}",
                        f"  carbs_g: {self._fmt_number(nutrition.get('carbs_g'))}",
                        f"  fat_g: {self._fmt_number(nutrition.get('fat_g'))}",
                    ]
                )
        else:
            lines.append("- Not available without depth image")

        if ratio_summary:
            lines.extend(["", "Ratio Summary"])
            for key, value in ratio_summary.items():
                lines.append(f"- {key}: {self._fmt_number(value)}")

        if missing:
            lines.extend(["", "Missing Requirements"])
            lines.extend(f"- {item}" for item in missing)

        if notes:
            lines.extend(["", "Notes"])
            lines.extend(f"- {note}" for note in notes)

        return "\n".join(lines)

    def _load_preview(self, file_path: str, target: str) -> None:
        try:
            with Image.open(file_path) as image:
                image.thumbnail((360, 220))
                photo = ImageTk.PhotoImage(image.copy())
        except Exception as exc:
            messagebox.showerror("Preview failed", f"Could not load preview: {exc}")
            return

        if target == "rgb":
            self._rgb_photo = photo
            self.rgb_preview.config(image=self._rgb_photo, text="")
        else:
            self._depth_photo = photo
            self.depth_preview.config(image=self._depth_photo, text="")

    def _update_confidence_label(self) -> None:
        self.confidence_label.config(text=f"{float(self.confidence_var.get()):.2f}")

    def _fmt_number(self, value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return str(value)

    def _fmt_percent(self, value: Any) -> str:
        if value is None:
            return "-"
        try:
            return f"{float(value) * 100:.1f}%"
        except (TypeError, ValueError):
            return str(value)

    def _set_text(self, widget: ScrolledText, text: str) -> None:
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.config(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    VisionAgentGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
