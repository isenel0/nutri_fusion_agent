"""General-purpose GUI for multimodal calorie and macro estimation."""

from __future__ import annotations

import asyncio
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any

from agents.decision.agent import FusionDecisionAgent
from agents.fusion.agent import FusionAgent
from schemas import AgentResponse
from agents.orchestrator.agent import OrchestratorAgent


async def run_multimodal_pipeline(
    image_path: str | None,
    depth_image_path: str | None,
    text_input: str | None,
    barcode_image_path: str | None,
) -> dict[str, Any]:
    """Run available modality agents in parallel and then fuse outputs.

    The pipeline is resilient: skipped or failed agents still produce visible
    placeholders so the GUI can always render a final section.
    """

    image_bytes: bytes | None = None
    image_filename: str | None = None
    depth_image_bytes: bytes | None = None
    depth_image_filename: str | None = None
    barcode_image_bytes: bytes | None = None
    barcode_image_filename: str | None = None

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

    if barcode_image_path:
        path = Path(barcode_image_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Barcode image path is invalid: {barcode_image_path}")
        barcode_image_bytes = path.read_bytes()
        barcode_image_filename = path.name

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

    if barcode_image_bytes is not None:
        payload["barcode_image_bytes"] = barcode_image_bytes
        payload["barcode_image_filename"] = barcode_image_filename

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
            "barcode_image_path": barcode_image_path,
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
        ttk.Label(barcode_row, text="Barcode Image", width=13).pack(side=tk.LEFT)
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

        self.vision_output = self._build_output_box(vision_tab)
        self.text_output = self._build_output_box(text_tab)
        self.barcode_output = self._build_output_box(barcode_tab)
        self.final_output = self._build_output_box(final_tab)

        self._set_text(self.vision_output, "Vision output will appear here.")
        self._set_text(self.text_output, "Text output will appear here.")
        self._set_text(self.barcode_output, "Barcode output will appear here.")
        self._set_text(
            self.final_output,
            "Final orchestrated output will appear here, even when some agents fail.",
        )

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

    def choose_barcode_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select barcode image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("All Files", "*.*"),
            ],
        )
        if file_path:
            self.barcode_image_path_var.set(file_path)

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

    def clear_inputs(self) -> None:
        self._run_generation += 1
        self.image_path_var.set("")
        self.depth_image_path_var.set("")
        self.barcode_image_path_var.set("")
        self.text_input_widget.delete("1.0", tk.END)
        self._reset_outputs()
        self.status_var.set("Ready")

    def _reset_outputs(self) -> None:
        self._set_text(self.vision_output, "Vision output will appear here.")
        self._set_text(self.text_output, "Text output will appear here.")
        self._set_text(self.barcode_output, "Barcode output will appear here.")
        self._set_text(
            self.final_output,
            "Final orchestrated output will appear here, even when some agents fail.",
        )

    def run_pipeline(self) -> None:
        image_path = self.image_path_var.get().strip() or None
        depth_image_path = self.depth_image_path_var.get().strip() or None
        text_input = self.text_input_widget.get("1.0", tk.END).strip() or None
        barcode_image_path = self.barcode_image_path_var.get().strip() or None

        if image_path is None and text_input is None and barcode_image_path is None:
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
            args=(run_generation, image_path, depth_image_path, text_input, barcode_image_path),
            daemon=True,
        )
        thread.start()

    def _run_pipeline_worker(
        self,
        run_generation: int,
        image_path: str | None,
        depth_image_path: str | None,
        text_input: str | None,
        barcode_image_path: str | None,
    ) -> None:
        try:
            result = asyncio.run(
                run_multimodal_pipeline(
                    image_path=image_path,
                    depth_image_path=depth_image_path,
                    text_input=text_input,
                    barcode_image_path=barcode_image_path,
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
        self._set_text(self.text_output, self._render_agent_block("text", outputs.get("text")))
        self._set_text(
            self.barcode_output,
            self._render_agent_block("barcode", outputs.get("barcode")),
        )

        final_rendered = self._render_final_block(final_output)
        self._set_text(self.final_output, final_rendered)
        self.status_var.set("Done")

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

    def _render_final_block(self, payload: dict[str, Any]) -> str:
        source = payload.get("source", "fusion")
        confidence = payload.get("confidence")
        error = payload.get("error")
        data = payload.get("data") or {}

        final_macros = data.get("final_macros") or {}
        calculation_ingredients = data.get("calculation_ingredients") or data.get("items") or []
        inputs_used = data.get("inputs_used") or {}
        decision = data.get("fusion_decision") or {}
        reasoning = data.get("reasoning_summary") or "-"

        lines = [
            f"Agent: {source}",
            f"Confidence: {self._fmt_number(confidence)}",
            f"Error: {error or '-'}",
            "",
            "Final Macros",
            f"- Calories (kcal): {self._fmt_number(final_macros.get('calories_kcal'))}",
            f"- Protein (g): {self._fmt_number(final_macros.get('protein_g'))}",
            f"- Carbs (g): {self._fmt_number(final_macros.get('carbs_g'))}",
            f"- Fat (g): {self._fmt_number(final_macros.get('fat_g'))}",
            "",
            "Calculation Ingredients",
            *self._render_calculation_ingredients(calculation_ingredients),
            "",
            "Inputs Used",
            f"- Vision: {bool(inputs_used.get('vision'))}",
            f"- Text: {bool(inputs_used.get('text'))}",
            f"- Barcode: {bool(inputs_used.get('barcode'))}",
            "",
            "Fusion Decision",
            f"- Primary source: {decision.get('primary_nutrition_source') or '-'}",
            f"- Mass source: {decision.get('mass_source') or '-'}",
            f"- Portion multiplier: {self._fmt_number(decision.get('portion_multiplier'))}",
            f"- Hidden ingredients: {', '.join(decision.get('hidden_ingredients') or []) or '-'}",
            "",
            f"Reasoning: {reasoning}",
        ]
        return "\n".join(lines)

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
