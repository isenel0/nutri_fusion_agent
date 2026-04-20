"""General-purpose GUI for multimodal calorie and macro estimation."""

from __future__ import annotations

import asyncio
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any

from agents.barcode.agent import BarcodeAgent
from agents.fusion.agent import FusionAgent
from agents.text.agent import TextAgent
from agents.vision.agent import VisionAgent
from schemas import AgentResponse


async def run_multimodal_pipeline(
    image_path: str | None,
    text_input: str | None,
    barcode_image_path: str | None,
) -> dict[str, Any]:
    """Run available modality agents in parallel and then fuse outputs.

    The pipeline is resilient: skipped or failed agents still produce visible
    placeholders so the GUI can always render a final section.
    """

    image_bytes: bytes | None = None
    image_filename: str | None = None
    barcode_image_bytes: bytes | None = None
    barcode_image_filename: str | None = None

    if image_path:
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Image path is invalid: {image_path}")
        image_bytes = path.read_bytes()
        image_filename = path.name

    if barcode_image_path:
        path = Path(barcode_image_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Barcode image path is invalid: {barcode_image_path}")
        barcode_image_bytes = path.read_bytes()
        barcode_image_filename = path.name

    vision_agent = VisionAgent()
    text_agent = TextAgent()
    barcode_agent = BarcodeAgent()
    fusion_agent = FusionAgent()

    tasks: dict[str, asyncio.Task[AgentResponse]] = {}

    if image_bytes is not None:
        vision_payload: dict[str, Any] = {
            "filename": image_filename,
            "content_type": None,
            "bytes": image_bytes,
        }
        tasks["vision"] = asyncio.create_task(vision_agent.process(vision_payload))

    if text_input:
        tasks["text"] = asyncio.create_task(text_agent.process(text_input))

    if barcode_image_bytes is not None:
        barcode_payload: dict[str, Any] = {
            "barcode": None,
            "image_bytes": barcode_image_bytes,
            "image_filename": barcode_image_filename,
        }
        tasks["barcode"] = asyncio.create_task(barcode_agent.process(barcode_payload))

    gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)

    outputs: dict[str, AgentResponse | None] = {
        "vision": None,
        "text": None,
        "barcode": None,
    }

    for key, result in zip(tasks.keys(), gathered):
        if isinstance(result, Exception):
            outputs[key] = AgentResponse(
                source=key,
                confidence=0.0,
                data={},
                error=f"{key} agent failed: {result}",
            )
        else:
            outputs[key] = result

    try:
        final_output = await fusion_agent.process(outputs)
    except Exception as exc:
        # Keep a stable final output shape even when fusion fails.
        final_output = AgentResponse(
            source="fusion",
            confidence=0.0,
            data={
                "final_macros": None,
                "reasoning_summary": "Fusion failed. Returning collected agent outputs only.",
                "inputs_used": {
                    "vision": outputs["vision"] is not None,
                    "text": outputs["text"] is not None,
                    "barcode": outputs["barcode"] is not None,
                },
            },
            error=f"fusion agent failed: {exc}",
        )

    return {
        "inputs": {
            "image_path": image_path,
            "text": text_input,
            "barcode_image_path": barcode_image_path,
        },
        "agent_outputs": {
            key: (value.model_dump() if value is not None else None)
            for key, value in outputs.items()
        },
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
        self.barcode_image_path_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")

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
        ttk.Label(image_row, text="Image").pack(side=tk.LEFT)
        ttk.Entry(image_row, textvariable=self.image_path_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=8
        )
        ttk.Button(image_row, text="Browse", command=self.choose_image).pack(side=tk.LEFT)

        barcode_row = ttk.Frame(left)
        barcode_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(barcode_row, text="Barcode Image").pack(side=tk.LEFT)
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

    def clear_inputs(self) -> None:
        self.image_path_var.set("")
        self.barcode_image_path_var.set("")
        self.text_input_widget.delete("1.0", tk.END)
        self.status_var.set("Ready")

    def run_pipeline(self) -> None:
        image_path = self.image_path_var.get().strip() or None
        text_input = self.text_input_widget.get("1.0", tk.END).strip() or None
        barcode_image_path = self.barcode_image_path_var.get().strip() or None

        if image_path is None and text_input is None and barcode_image_path is None:
            self.status_var.set(
                "Please provide at least one input: meal image, text, or barcode image."
            )
            return

        self.status_var.set("Running agents in parallel...")

        thread = threading.Thread(
            target=self._run_pipeline_worker,
            args=(image_path, text_input, barcode_image_path),
            daemon=True,
        )
        thread.start()

    def _run_pipeline_worker(
        self,
        image_path: str | None,
        text_input: str | None,
        barcode_image_path: str | None,
    ) -> None:
        try:
            result = asyncio.run(
                run_multimodal_pipeline(
                    image_path=image_path,
                    text_input=text_input,
                    barcode_image_path=barcode_image_path,
                )
            )
        except Exception as exc:
            self.root.after(0, lambda: self.status_var.set(f"Pipeline failed: {exc}"))
            return

        self.root.after(0, lambda: self._render_results(result))

    def _render_results(self, payload: dict[str, Any]) -> None:
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

        # Vision and text agents currently expose generic macros payload.
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
        inputs_used = data.get("inputs_used") or {}
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
            "Inputs Used",
            f"- Vision: {bool(inputs_used.get('vision'))}",
            f"- Text: {bool(inputs_used.get('text'))}",
            f"- Barcode: {bool(inputs_used.get('barcode'))}",
            "",
            f"Reasoning: {reasoning}",
        ]
        return "\n".join(lines)

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
