"""General-purpose GUI for multimodal calorie and macro estimation."""

from __future__ import annotations

import asyncio
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any

from PIL import Image, ImageTk

from agents.decision.agent import FusionDecisionAgent
from agents.fusion.agent import FusionAgent
from schemas import AgentResponse
from agents.orchestrator.agent import OrchestratorAgent


async def run_multimodal_pipeline(
    image_path: str | None,
    depth_image_path: str | None,
    text_input: str | None,
    barcode_image_path: str | None = None,
    barcode_image_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Run available modality agents in parallel and then fuse outputs.

    The pipeline is resilient: skipped or failed agents still produce visible
    placeholders so the GUI can always render a final section.
    """

    image_bytes: bytes | None = None
    image_filename: str | None = None
    depth_image_bytes: bytes | None = None
    depth_image_filename: str | None = None
    barcode_image_payloads: list[dict[str, Any]] = []

    if image_path:
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Image path is invalid: {image_path}")
        image_bytes = path.read_bytes()
        image_filename = path.name

    if depth_image_path:
        path = Path(depth_image_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Depth image path is invalid: {depth_image_path}")
        depth_image_bytes = path.read_bytes()
        depth_image_filename = path.name

    selected_barcode_paths = barcode_image_paths or (
        [barcode_image_path] if barcode_image_path else []
    )
    for barcode_path in selected_barcode_paths:
        path = Path(barcode_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Barcode image path is invalid: {barcode_path}")
        barcode_image_payloads.append(
            {
                "image_bytes": path.read_bytes(),
                "image_filename": path.name,
            }
        )

    if barcode_image_path and not selected_barcode_paths:
        path = Path(barcode_image_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Barcode image path is invalid: {barcode_image_path}")

    orchestrator = OrchestratorAgent()
    decision_agent = FusionDecisionAgent()
    fusion_agent = FusionAgent()

    payload: dict[str, Any] = {
        "text": text_input,
    }

    if image_bytes is not None:
        payload["image_bytes"] = image_bytes
        payload["image_filename"] = image_filename

    if depth_image_bytes is not None:
        payload["depth_bytes"] = depth_image_bytes
        payload["depth_image_filename"] = depth_image_filename

    if barcode_image_payloads:
        if len(barcode_image_payloads) == 1:
            payload["barcode_image_bytes"] = barcode_image_payloads[0]["image_bytes"]
            payload["barcode_image_filename"] = barcode_image_payloads[0]["image_filename"]
        else:
            payload["barcode_image_payloads"] = barcode_image_payloads

    orchestrator_result = await orchestrator.process(payload)
    decision_result = await decision_agent.process(orchestrator_result)

    try:
        final_output = await fusion_agent.process(
            {
                "orchestrator": orchestrator_result,
                "decision": decision_result,
            }
        )
    except Exception as exc:
        # Keep a stable final output shape even when fusion fails.
        final_output = AgentResponse(
            source="fusion",
            confidence=0.0,
            data={
                "final_macros": None,
                "reasoning_summary": "Fusion failed. Returning collected agent outputs only.",
                "inputs_used": {},
            },
            error=f"fusion agent failed: {exc}",
        )

    outputs = (orchestrator_result.data.get("outputs") or {})

    return {
        "inputs": {
            "image_path": image_path,
            "depth_image_path": depth_image_path,
            "text": text_input,
            "barcode_image_paths": selected_barcode_paths,
        },
        "orchestrator_output": orchestrator_result.model_dump(),
        "decision_output": decision_result.model_dump(),
        "agent_outputs": outputs,
        "final_output": final_output.model_dump(),
    }


class MultiAgentGUI:
    """Desktop UI for manual testing of multimodal agent orchestration."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Nutri Fusion - Multi-Agent Test GUI")
        self.root.geometry("1140x760")
        self.root.minsize(980, 620)

        self.image_path_var = tk.StringVar(value="")
        self.depth_image_path_var = tk.StringVar(value="")
        self.barcode_image_path_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")
        self._run_generation = 0

        self.text_input_widget: tk.Text
        self.vision_preview_frame: ttk.Frame
        self.barcode_preview_frame: ttk.Frame
        self.final_preview_frame: ttk.Frame
        self.final_preview_canvas: tk.Canvas
        self.final_popup_button: ttk.Button
        self._last_final_rendered = "Final output will appear here."
        self._last_final_output: dict[str, Any] = {}
        self._preview_images: list[ImageTk.PhotoImage] = []
        self.vision_output: ScrolledText
        self.text_output: ScrolledText
        self.barcode_output: ScrolledText
        self.final_output: ScrolledText

        self._build_layout()

    def _build_layout(self) -> None:
        container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = ttk.Frame(container, padding=10)
        right = ttk.Frame(container, padding=10)

        container.add(left, weight=1)
        container.add(right, weight=2)

        ttk.Label(left, text="Inputs", font=("Helvetica", 14, "bold")).pack(
            anchor="w", pady=(0, 10)
        )

        image_row = ttk.Frame(left)
        image_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(image_row, text="RGB Image", width=13).pack(side=tk.LEFT)
        ttk.Entry(image_row, textvariable=self.image_path_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=8
        )
        ttk.Button(image_row, text="Browse", command=self.choose_image).pack(side=tk.LEFT)

        depth_row = ttk.Frame(left)
        depth_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(depth_row, text="Depth Image", width=13).pack(side=tk.LEFT)
        ttk.Entry(depth_row, textvariable=self.depth_image_path_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=8
        )
        ttk.Button(depth_row, text="Browse", command=self.choose_depth_image).pack(
            side=tk.LEFT
        )

        barcode_row = ttk.Frame(left)
        barcode_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(barcode_row, text="Barcodes", width=13).pack(side=tk.LEFT)
        ttk.Entry(barcode_row, textvariable=self.barcode_image_path_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=8
        )
        ttk.Button(barcode_row, text="Browse", command=self.choose_barcode_image).pack(
            side=tk.LEFT
        )

        ttk.Label(left, text="Meal Text").pack(anchor="w")
        self.text_input_widget = tk.Text(left, height=8, wrap=tk.WORD)
        self.text_input_widget.pack(fill=tk.BOTH, expand=False, pady=(4, 10))

        button_row = ttk.Frame(left)
        button_row.pack(fill=tk.X)
        ttk.Button(button_row, text="Run Pipeline", command=self.run_pipeline).pack(
            side=tk.LEFT
        )
        ttk.Button(button_row, text="Clear", command=self.clear_inputs).pack(
            side=tk.LEFT, padx=8
        )

        ttk.Label(left, textvariable=self.status_var, foreground="#0a4b78").pack(
            anchor="w", pady=(10, 0)
        )

        ttk.Label(right, text="Outputs", font=("Helvetica", 14, "bold")).pack(
            anchor="w", pady=(0, 8)
        )

        tabs = ttk.Notebook(right)
        tabs.pack(fill=tk.BOTH, expand=True)

        vision_tab = ttk.Frame(tabs)
        text_tab = ttk.Frame(tabs)
        barcode_tab = ttk.Frame(tabs)
        final_tab = ttk.Frame(tabs)

        tabs.add(vision_tab, text="Vision")
        tabs.add(text_tab, text="Text")
        tabs.add(barcode_tab, text="Barcode")
        tabs.add(final_tab, text="Final")

        self.vision_preview_frame, self.vision_output = self._build_output_panel(
            vision_tab,
            "Input Images",
        )
        self.text_output = self._build_output_box(text_tab)
        self.barcode_preview_frame, self.barcode_output = self._build_output_panel(
            barcode_tab,
            "Barcode Images",
        )
        self.final_preview_frame, self.final_output = self._build_output_panel(
            final_tab,
            "Selected Images",
            scroll_preview=True,
            popup_button=True,
        )

        self._set_text(self.vision_output, "Vision output will appear here.")
        self._set_text(self.text_output, "Text output will appear here.")
        self._set_text(self.barcode_output, "Barcode output will appear here.")
        self._set_text(
            self.final_output,
            "Final orchestrated output will appear here, even when some agents fail.",
        )
        self._update_previews()

    def _build_output_panel(
        self,
        parent: ttk.Frame,
        preview_title: str,
        scroll_preview: bool = False,
        popup_button: bool = False,
    ) -> tuple[ttk.Frame, ScrolledText]:
        panel = ttk.Frame(parent)
        panel.pack(fill=tk.BOTH, expand=True)

        preview_container = ttk.Frame(panel, padding=(0, 0, 0, 8))
        preview_container.pack(fill=tk.X)
        ttk.Label(
            preview_container,
            text=preview_title,
            font=("Helvetica", 11, "bold"),
        ).pack(anchor="w", pady=(0, 6))

        if scroll_preview:
            preview_canvas = tk.Canvas(preview_container, height=290, highlightthickness=0)
            preview_scrollbar = ttk.Scrollbar(
                preview_container,
                orient=tk.HORIZONTAL,
                command=preview_canvas.xview,
            )
            preview_canvas.configure(xscrollcommand=preview_scrollbar.set)
            preview_canvas.pack(fill=tk.X, expand=False)
            preview_scrollbar.pack(fill=tk.X, pady=(4, 0))

            preview_frame = ttk.Frame(preview_canvas)
            window_id = preview_canvas.create_window(
                (0, 0),
                window=preview_frame,
                anchor="nw",
            )

            def _sync_scrollregion(_event: tk.Event) -> None:
                preview_canvas.configure(scrollregion=preview_canvas.bbox("all"))

            def _sync_height(event: tk.Event) -> None:
                preview_canvas.itemconfigure(window_id, height=event.height)

            preview_frame.bind("<Configure>", _sync_scrollregion)
            preview_canvas.bind("<Configure>", _sync_height)
            self.final_preview_canvas = preview_canvas
        else:
            preview_frame = ttk.Frame(preview_container)
            preview_frame.pack(fill=tk.X)

        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(4, 8))
        if popup_button:
            controls = ttk.Frame(panel)
            controls.pack(fill=tk.X, pady=(0, 8))
            self.final_popup_button = ttk.Button(
                controls,
                text="View Full Final Output",
                command=self._open_final_popup,
            )
            self.final_popup_button.pack(side=tk.LEFT)

        output = self._build_output_box(panel)
        return preview_frame, output

    def _build_output_box(self, parent: ttk.Frame) -> ScrolledText:
        box = ScrolledText(parent, wrap=tk.WORD, font=("Menlo", 11), height=20)
        box.pack(fill=tk.BOTH, expand=True)
        box.config(state=tk.DISABLED)
        return box

    def choose_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select meal image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("All Files", "*.*"),
            ],
        )
        if file_path:
            self.image_path_var.set(file_path)
            self._update_previews()

    def choose_barcode_image(self) -> None:
        file_paths = filedialog.askopenfilenames(
            title="Select barcode image(s)",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("All Files", "*.*"),
            ],
        )
        if file_paths:
            self.barcode_image_path_var.set("; ".join(file_paths))
            self._update_previews()

    def choose_depth_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select depth image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
                ("All Files", "*.*"),
            ],
        )
        if file_path:
            self.depth_image_path_var.set(file_path)
            self._update_previews()

    def clear_inputs(self) -> None:
        self._run_generation += 1
        self.image_path_var.set("")
        self.depth_image_path_var.set("")
        self.barcode_image_path_var.set("")
        self.text_input_widget.delete("1.0", tk.END)
        self._reset_outputs()
        self._update_previews()
        self.status_var.set("Ready")

    def _reset_outputs(self) -> None:
        self._set_text(self.vision_output, "Vision output will appear here.")
        self._set_text(self.text_output, "Text output will appear here.")
        self._set_text(self.barcode_output, "Barcode output will appear here.")
        self._set_text(
            self.final_output,
            "Final orchestrated output will appear here, even when some agents fail.",
        )
        self._last_final_rendered = "Final orchestrated output will appear here, even when some agents fail."

    def run_pipeline(self) -> None:
        image_path = self.image_path_var.get().strip() or None
        depth_image_path = self.depth_image_path_var.get().strip() or None
        text_input = self.text_input_widget.get("1.0", tk.END).strip() or None
        barcode_image_paths = self._barcode_image_paths()
        self._update_previews()

        if image_path is None and text_input is None and not barcode_image_paths:
            self.status_var.set(
                "Please provide at least one input: meal image, text, or barcode image."
            )
            return

        if depth_image_path is not None and image_path is None:
            self.status_var.set("Depth image requires an RGB image.")
            return

        self.status_var.set("Running orchestrator, local LLM decision, and fusion...")
        self._run_generation += 1
        run_generation = self._run_generation

        thread = threading.Thread(
            target=self._run_pipeline_worker,
            args=(run_generation, image_path, depth_image_path, text_input, barcode_image_paths),
            daemon=True,
        )
        thread.start()

    def _barcode_image_paths(self) -> list[str]:
        raw_value = self.barcode_image_path_var.get().strip()
        if not raw_value:
            return []
        return [
            item.strip()
            for item in raw_value.split(";")
            if item.strip()
        ]

    def _update_previews(self) -> None:
        if not hasattr(self, "vision_preview_frame"):
            return

        for frame in (
            self.vision_preview_frame,
            self.barcode_preview_frame,
            self.final_preview_frame,
        ):
            for child in frame.winfo_children():
                child.destroy()
        self._preview_images = []

        vision_specs: list[tuple[str, str]] = []
        barcode_specs: list[tuple[str, str]] = []
        final_specs: list[tuple[str, str]] = []
        image_path = self.image_path_var.get().strip()
        depth_path = self.depth_image_path_var.get().strip()
        if image_path:
            vision_specs.append(("RGB Image", image_path))
            final_specs.append(("RGB Image", image_path))
        if depth_path:
            vision_specs.append(("Depth Image", depth_path))
            final_specs.append(("Depth Image", depth_path))
        for index, barcode_path in enumerate(self._barcode_image_paths(), start=1):
            barcode_specs.append((f"Barcode Image {index}", barcode_path))
            final_specs.append((f"Barcode Image {index}", barcode_path))

        self._render_preview_specs(
            self.vision_preview_frame,
            vision_specs,
            "RGB/depth previews will appear here.",
        )
        self._render_preview_specs(
            self.barcode_preview_frame,
            barcode_specs,
            "Barcode image previews will appear here.",
        )
        self._render_preview_specs(
            self.final_preview_frame,
            final_specs,
            "Selected image previews will appear here.",
        )

    def _render_preview_specs(
        self,
        parent: ttk.Frame,
        preview_specs: list[tuple[str, str]],
        empty_text: str,
    ) -> None:
        if not preview_specs:
            ttk.Label(parent, text=empty_text, foreground="#555555").pack(anchor="w")
            return

        for title, path in preview_specs:
            self._add_preview_image(parent, title, path)

    def _add_preview_image(self, parent: ttk.Frame, title: str, path: str) -> None:
        section = ttk.Frame(parent, padding=(0, 0, 10, 8))
        section.pack(side=tk.LEFT, anchor="n")

        ttk.Label(section, text=title, font=("Helvetica", 12, "bold")).pack(anchor="w")
        ttk.Label(section, text=Path(path).name, foreground="#555555", wraplength=320).pack(
            anchor="w",
            pady=(2, 6),
        )

        try:
            with Image.open(path) as image:
                preview = image.convert("RGB")
                preview.thumbnail((340, 230))
                photo = ImageTk.PhotoImage(preview)
        except Exception as exc:
            ttk.Label(
                section,
                text=f"Preview unavailable: {exc}",
                foreground="#9b1c1c",
            ).pack(anchor="w")
            return

        self._preview_images.append(photo)
        frame = ttk.Frame(section, relief=tk.SOLID, borderwidth=1, padding=6)
        frame.pack(anchor="w")
        ttk.Label(frame, image=photo).pack()

    def _run_pipeline_worker(
        self,
        run_generation: int,
        image_path: str | None,
        depth_image_path: str | None,
        text_input: str | None,
        barcode_image_paths: list[str],
    ) -> None:
        try:
            result = asyncio.run(
                run_multimodal_pipeline(
                    image_path=image_path,
                    depth_image_path=depth_image_path,
                    text_input=text_input,
                    barcode_image_paths=barcode_image_paths,
                )
            )
        except Exception as exc:
            self.root.after(
                0,
                lambda: self._set_failure_status(run_generation, exc),
            )
            return

        self.root.after(0, lambda: self._render_results(run_generation, result))

    def _set_failure_status(self, run_generation: int, exc: Exception) -> None:
        if run_generation == self._run_generation:
            self.status_var.set(f"Pipeline failed: {exc}")

    def _render_results(self, run_generation: int, payload: dict[str, Any]) -> None:
        if run_generation != self._run_generation:
            return

        outputs = payload.get("agent_outputs", {})
        final_output = payload.get("final_output", {})

        self._set_text(self.vision_output, self._render_agent_block("vision", outputs.get("vision")))
        text_blocks = [
            self._render_input_text(payload),
            "",
            self._render_agent_block("text", outputs.get("text")),
            "",
            self._render_agent_block("text_nutrition", outputs.get("text_nutrition")),
        ]
        self._set_text(self.text_output, "\n".join(text_blocks))
        self._set_text(
            self.barcode_output,
            self._render_agent_block("barcode", outputs.get("barcode")),
        )

        final_rendered = self._render_final_block(final_output)
        self._last_final_rendered = final_rendered
        self._last_final_output = final_output
        self._set_text(self.final_output, final_rendered)
        self.status_var.set("Done")

    def _open_final_popup(self) -> None:
        popup = tk.Toplevel(self.root)
        popup.title("Full Final Output")
        popup.geometry("1020x780")
        popup.minsize(820, 620)

        header = ttk.Frame(popup, padding=(10, 10, 10, 6))
        header.pack(fill=tk.X)
        ttk.Label(
            header,
            text="Full Final Output",
            font=("Helvetica", 14, "bold"),
        ).pack(side=tk.LEFT)
        ttk.Button(header, text="Close", command=popup.destroy).pack(side=tk.RIGHT)

        self._build_final_charts(popup, self._last_final_output)

        box = ScrolledText(popup, wrap=tk.WORD, font=("Menlo", 12), padx=12, pady=12)
        box.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        box.insert(tk.END, self._last_final_rendered)
        box.config(state=tk.DISABLED)

    def _build_final_charts(self, parent: tk.Toplevel, payload: dict[str, Any]) -> None:
        data = payload.get("data") or {}
        final_macros = data.get("final_macros") or {}
        ingredients = data.get("calculation_ingredients") or data.get("items") or []

        visual = ttk.Frame(parent, padding=(10, 0, 10, 10))
        visual.pack(fill=tk.X)

        cards = ttk.Frame(visual)
        cards.pack(fill=tk.X, pady=(0, 10))
        self._summary_card(
            cards,
            "Calories",
            f"{self._fmt_number(final_macros.get('calories_kcal'))} kcal",
            "#e85d04",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self._summary_card(
            cards,
            "Mass",
            f"{self._fmt_number(final_macros.get('mass_g'))} g",
            "#2a9d8f",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self._summary_card(
            cards,
            "Protein",
            f"{self._fmt_number(final_macros.get('protein_g'))} g",
            "#457b9d",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self._summary_card(
            cards,
            "Carbs / Fat",
            (
                f"{self._fmt_number(final_macros.get('carbs_g'))}g / "
                f"{self._fmt_number(final_macros.get('fat_g'))}g"
            ),
            "#7b2cbf",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        chart_row = ttk.Frame(visual)
        chart_row.pack(fill=tk.X)

        macro_panel = ttk.Frame(chart_row)
        macro_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        ttk.Label(
            macro_panel,
            text="Macronutrient Distribution",
            font=("Helvetica", 12, "bold"),
        ).pack(anchor="w")
        macro_canvas = tk.Canvas(macro_panel, height=150, bg="#ffffff", highlightthickness=1, highlightbackground="#d0d7de")
        macro_canvas.pack(fill=tk.X, pady=(6, 0))
        self._draw_macro_chart(macro_canvas, final_macros)

        ingredient_panel = ttk.Frame(chart_row)
        ingredient_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(
            ingredient_panel,
            text="Ingredient Calories",
            font=("Helvetica", 12, "bold"),
        ).pack(anchor="w")
        ingredient_canvas = tk.Canvas(ingredient_panel, height=150, bg="#ffffff", highlightthickness=1, highlightbackground="#d0d7de")
        ingredient_canvas.pack(fill=tk.X, pady=(6, 0))
        self._draw_ingredient_chart(ingredient_canvas, ingredients)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=(0, 8))

    def _summary_card(
        self,
        parent: ttk.Frame,
        label: str,
        value: str,
        color: str,
    ) -> tk.Frame:
        card = tk.Frame(parent, bg=color, padx=12, pady=10)
        tk.Label(
            card,
            text=label,
            bg=color,
            fg="#ffffff",
            font=("Helvetica", 10, "bold"),
        ).pack(anchor="w")
        tk.Label(
            card,
            text=value,
            bg=color,
            fg="#ffffff",
            font=("Helvetica", 16, "bold"),
        ).pack(anchor="w", pady=(4, 0))
        return card

    def _draw_macro_chart(self, canvas: tk.Canvas, macros: dict[str, Any]) -> None:
        values = [
            ("Protein", self._to_float(macros.get("protein_g")), "#457b9d"),
            ("Carbs", self._to_float(macros.get("carbs_g")), "#f4a261"),
            ("Fat", self._to_float(macros.get("fat_g")), "#7b2cbf"),
        ]

        def draw() -> None:
            canvas.delete("all")
            width = max(canvas.winfo_width(), 320)
            max_value = max((value for _, value, _ in values), default=0.0)
            if max_value <= 0:
                canvas.create_text(16, 16, anchor="nw", text="No macro values available.", fill="#555555")
                return

            left = 92
            right = width - 24
            bar_width = max(right - left, 80)
            y = 24
            for label, value, color in values:
                bar_len = (value / max_value) * bar_width if max_value else 0
                canvas.create_text(14, y + 10, anchor="w", text=label, fill="#222222", font=("Helvetica", 10, "bold"))
                canvas.create_rectangle(left, y, right, y + 20, fill="#eef1f4", outline="")
                canvas.create_rectangle(left, y, left + bar_len, y + 20, fill=color, outline="")
                canvas.create_text(
                    right,
                    y + 10,
                    anchor="e",
                    text=f"{self._fmt_number(value)} g",
                    fill="#222222",
                    font=("Helvetica", 10),
                )
                y += 38

        canvas.after_idle(draw)
        canvas.bind("<Configure>", lambda _event: draw())

    def _draw_ingredient_chart(self, canvas: tk.Canvas, ingredients: list[dict[str, Any]]) -> None:
        rows = []
        for ingredient in ingredients[:5]:
            nutrition = ingredient.get("nutrition") or {}
            calories = self._to_float(nutrition.get("calories_kcal"))
            rows.append((str(ingredient.get("name") or "unknown"), calories))

        def draw() -> None:
            canvas.delete("all")
            width = max(canvas.winfo_width(), 320)
            if not rows:
                canvas.create_text(16, 16, anchor="nw", text="No ingredients available.", fill="#555555")
                return

            max_value = max((value for _, value in rows), default=0.0)
            if max_value <= 0:
                canvas.create_text(16, 16, anchor="nw", text="No ingredient calories available.", fill="#555555")
                return

            left = 132
            right = width - 24
            bar_width = max(right - left, 80)
            y = 14
            colors = ["#e85d04", "#2a9d8f", "#457b9d", "#7b2cbf", "#6c757d"]
            for index, (name, value) in enumerate(rows):
                display_name = name if len(name) <= 18 else name[:17] + "…"
                bar_len = (value / max_value) * bar_width if max_value else 0
                color = colors[index % len(colors)]
                canvas.create_text(12, y + 10, anchor="w", text=display_name, fill="#222222", font=("Helvetica", 10, "bold"))
                canvas.create_rectangle(left, y, right, y + 20, fill="#eef1f4", outline="")
                canvas.create_rectangle(left, y, left + bar_len, y + 20, fill=color, outline="")
                canvas.create_text(
                    right,
                    y + 10,
                    anchor="e",
                    text=f"{self._fmt_number(value)} kcal",
                    fill="#222222",
                    font=("Helvetica", 10),
                )
                y += 26

        canvas.after_idle(draw)
        canvas.bind("<Configure>", lambda _event: draw())

    def _render_agent_block(self, agent_name: str, payload: dict[str, Any] | None) -> str:
        if payload is None:
            return (
                f"Agent: {agent_name}\n"
                "Status: skipped\n"
                "Reason: No compatible input provided."
            )

        source = payload.get("source", agent_name)
        confidence = payload.get("confidence")
        error = payload.get("error")
        data = payload.get("data") or {}

        lines = [
            f"Agent: {source}",
            f"Confidence: {self._fmt_number(confidence)}",
            f"Error: {error or '-'}",
            "",
        ]

        if source == "barcode":
            product = data.get("product") or {}
            barcode_value = (data.get("barcode") or {}).get("value")
            macros = data.get("nutrition_per_100g") or {}
            candidates = data.get("candidates") or []

            lines.extend(
                [
                    "Product",
                    f"- Barcode: {barcode_value or '-'}",
                    f"- Name: {product.get('name') or '-'}",
                    f"- Brand: {product.get('brand') or '-'}",
                    "",
                    "Per 100g Macros",
                    f"- Calories (kcal): {self._fmt_number(macros.get('calories_kcal'))}",
                    f"- Protein (g): {self._fmt_number(macros.get('protein_g'))}",
                    f"- Carbs (g): {self._fmt_number(macros.get('carbs_g'))}",
                    f"- Fat (g): {self._fmt_number(macros.get('fat_g'))}",
                ]
            )
            if candidates:
                lines.extend(["", "Barcode Candidates"])
                for candidate in candidates:
                    candidate_data = candidate.get("data") or {}
                    candidate_product = candidate_data.get("product") or {}
                    candidate_barcode = (candidate_data.get("barcode") or {}).get("value")
                    candidate_nutrition = candidate_data.get("nutrition_per_100g") or {}
                    lines.append(
                        "- "
                        + f"{candidate_data.get('input_filename') or '-'}"
                        + f": barcode {candidate_barcode or '-'}, "
                        + f"{candidate_product.get('name') or '-'}, "
                        + f"{self._fmt_number(candidate_nutrition.get('calories_kcal'))} kcal/100g, "
                        + f"error: {candidate.get('error') or '-'}"
                    )
            return "\n".join(lines)

        if source == "vision":
            totals = data.get("totals") or {}
            detected = data.get("detected_items") or []
            detected_text = ", ".join(
                f"{item.get('name')} ({self._fmt_number(item.get('pixel_ratio'))})"
                for item in detected[:8]
            )
            lines.extend(
                [
                    f"Mode: {data.get('mode') or '-'}",
                    f"Detected: {detected_text or '-'}",
                    "",
                    "Vision Totals",
                    f"- Calories (kcal): {self._fmt_number(totals.get('calories_kcal'))}",
                    f"- Mass (g): {self._fmt_number(totals.get('mass_g'))}",
                    f"- Protein (g): {self._fmt_number(totals.get('protein_g'))}",
                    f"- Carbs (g): {self._fmt_number(totals.get('carbs_g'))}",
                    f"- Fat (g): {self._fmt_number(totals.get('fat_g'))}",
                ]
            )
            return "\n".join(lines)

        if source == "text":
            parsed = data.get("parsed") or {}
            lines.extend(
                [
                    f"Foods: {', '.join(parsed.get('foods') or []) or '-'}",
                    f"Quantities: {', '.join(parsed.get('quantities') or []) or '-'}",
                    f"Units: {', '.join(parsed.get('units') or []) or '-'}",
                    f"Methods: {', '.join(parsed.get('methods') or []) or '-'}",
                    f"Modifiers: {', '.join(parsed.get('modifiers') or []) or '-'}",
                    f"Portion multiplier: {self._fmt_number(parsed.get('portion_multiplier'))}",
                ]
            )
            return "\n".join(lines)

        if source == "text_nutrition":
            totals = data.get("totals") or {}
            items = data.get("items") or []
            lines.extend(
                [
                    "Text Nutrition Totals",
                    f"- Calories (kcal): {self._fmt_number(totals.get('calories_kcal'))}",
                    f"- Mass (g): {self._fmt_number(totals.get('mass_g'))}",
                    f"- Protein (g): {self._fmt_number(totals.get('protein_g'))}",
                    f"- Carbs (g): {self._fmt_number(totals.get('carbs_g'))}",
                    f"- Fat (g): {self._fmt_number(totals.get('fat_g'))}",
                    "",
                    "Text Nutrition Items",
                ]
            )
            for item in items:
                nutrition = item.get("nutrition") or {}
                lines.append(
                    "- "
                    + f"{item.get('name') or '-'}"
                    + f" -> {item.get('lookup_name') or '-'}"
                    + f": {self._fmt_number(item.get('mass_g'))}g"
                    + f" ({item.get('mass_source') or '-'})"
                    + f", {self._fmt_number(nutrition.get('calories_kcal'))} kcal"
                )
            return "\n".join(lines)

        macros = data.get("macros") or {}
        lines.extend(
            [
                "Meal",
                f"- Name: {data.get('food_name') or '-'}",
                f"- Portion: {data.get('portion_size') or '-'}",
                "",
                "Macros",
                f"- Calories (kcal): {self._fmt_number(macros.get('calories_kcal'))}",
                f"- Protein (g): {self._fmt_number(macros.get('protein_g'))}",
                f"- Carbs (g): {self._fmt_number(macros.get('carbs_g'))}",
                f"- Fat (g): {self._fmt_number(macros.get('fat_g'))}",
            ]
        )
        return "\n".join(lines)

    def _render_input_text(self, payload: dict[str, Any]) -> str:
        inputs = payload.get("inputs") or {}
        text = inputs.get("text")
        return (
            "Input Text\n"
            f"- {text or '-'}"
        )

    def _render_final_block(self, payload: dict[str, Any]) -> str:
        source = payload.get("source", "fusion")
        confidence = payload.get("confidence")
        error = payload.get("error")
        data = payload.get("data") or {}

        final_macros = data.get("final_macros") or {}
        calculation_ingredients = data.get("calculation_ingredients") or data.get("items") or []
        inputs_used = data.get("inputs_used") or {}
        decision = data.get("fusion_decision") or {}
        agent_outputs = data.get("agent_outputs") or {}
        resolution = ((agent_outputs.get("ingredient_resolution") or {}).get("data") or {})
        reasoning = data.get("reasoning_summary") or "-"

        lines = [
            "============================================================",
            "FINAL MEAL ANALYSIS",
            "============================================================",
            f"Agent: {source}    Confidence: {self._fmt_number(confidence)}",
            f"Error: {error or '-'}",
            "",
            "TOTALS",
            "------",
            f"Calories : {self._fmt_number(final_macros.get('calories_kcal'))} kcal",
            f"Mass     : {self._fmt_number(final_macros.get('mass_g'))} g",
            "",
            "MACRONUTRIENTS",
            "--------------",
            f"Protein  : {self._fmt_number(final_macros.get('protein_g'))} g",
            f"Carbs    : {self._fmt_number(final_macros.get('carbs_g'))} g",
            f"Fat      : {self._fmt_number(final_macros.get('fat_g'))} g",
            "",
            "CALCULATION INGREDIENTS",
            "-----------------------",
            *self._render_calculation_ingredients(calculation_ingredients),
            "",
            "INPUTS USED",
            "-----------",
            f"- Vision: {bool(inputs_used.get('vision'))}",
            f"- Depth: {bool(inputs_used.get('depth'))}",
            f"- Text: {bool(inputs_used.get('text'))}",
            f"- Text nutrition: {bool(inputs_used.get('text_nutrition'))}",
            f"- Ingredient resolution: {bool(inputs_used.get('ingredient_resolution'))}",
            f"- Barcode: {bool(inputs_used.get('barcode'))}",
            "",
            "INGREDIENT RESOLUTION",
            "---------------------",
            *self._render_resolution_summary(resolution),
            "",
            "FUSION DECISION",
            "---------------",
            f"- Primary source: {decision.get('primary_nutrition_source') or '-'}",
            f"- Mass source: {decision.get('mass_source') or '-'}",
            f"- Portion multiplier: {self._fmt_number(decision.get('portion_multiplier'))}",
            f"- Hidden ingredients: {', '.join(decision.get('hidden_ingredients') or []) or '-'}",
            "",
            "REASONING",
            "---------",
            reasoning,
        ]
        return "\n".join(lines)

    def _render_resolution_summary(self, resolution: dict[str, Any]) -> list[str]:
        if not resolution:
            return ["- -"]
        lines = []
        replacements = resolution.get("replacements") or {}
        if replacements:
            lines.append(
                "- Replacements: "
                + ", ".join(f"{old} -> {new}" for old, new in replacements.items())
            )
        assumptions = resolution.get("assumptions") or []
        if assumptions:
            lines.extend(f"- {assumption}" for assumption in assumptions[:4])
        if not lines:
            lines.append("- No replacements.")
        return lines

    def _render_calculation_ingredients(self, ingredients: list[dict[str, Any]]) -> list[str]:
        if not ingredients:
            return ["- -"]

        lines = []
        for ingredient in ingredients:
            nutrition = ingredient.get("nutrition") or {}
            lines.append(
                "- "
                + f"{ingredient.get('name') or 'unknown'}"
                + f" ({ingredient.get('source') or '-'})"
                + f": {self._fmt_number(nutrition.get('mass_g') or ingredient.get('mass_g'))}g,"
                + f" {self._fmt_number(nutrition.get('calories_kcal'))} kcal,"
                + f" P {self._fmt_number(nutrition.get('protein_g'))}g,"
                + f" C {self._fmt_number(nutrition.get('carbs_g'))}g,"
                + f" F {self._fmt_number(nutrition.get('fat_g'))}g"
            )
        return lines

    def _fmt_number(self, value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.2f}".rstrip("0").rstrip(".")
        return str(value)

    def _to_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _set_text(self, widget: ScrolledText, text: str) -> None:
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.config(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    app = MultiAgentGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
