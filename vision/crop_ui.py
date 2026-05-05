"""crop_ui.py — NiceGUI front-end for the ROI cropper.

Usage:
    python vision/crop_ui.py
    # then open http://localhost:8080

Workflow:
    1. Upload an example image.
    2. Click the top-left corner, then the bottom-right corner. The
       chosen rectangle is drawn as an SVG overlay.
    3. Type the input and output directories.
    4. Press "Crop all" — every image under input is cropped to the
       same box and written, structure-preserving, under output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from nicegui import events, ui

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision.crop_to_roi import Roi, crop_directory


class CropApp:
    def __init__(self) -> None:
        self.example: np.ndarray | None = None
        self.example_data_url: str = ""
        self.corners: list[tuple[int, int]] = []
        self.roi: Roi | None = None

        with ui.column().classes("w-full max-w-3xl mx-auto gap-4 p-4"):
            ui.label("ROI Cropper").classes("text-2xl font-bold")
            ui.label(
                "Upload an example, click two opposite corners to set the ROI, "
                "then run on a directory."
            ).classes("text-sm text-gray-600")

            ui.upload(
                label="Example image",
                auto_upload=True,
                on_upload=self._on_upload,
            ).props("accept=image/*").classes("w-full")

            self.image = ui.interactive_image(
                "",
                on_mouse=self._on_click,
                events=["mousedown"],
                cross=True,
            ).classes("w-full")

            self.roi_label = ui.label("ROI: (click two corners on the example)")

            with ui.row().classes("w-full gap-2"):
                ui.button("Reset corners", on_click=self._reset).props("flat")

            self.input_dir = ui.input("Input directory").classes("w-full")
            self.output_dir = ui.input("Output directory").classes("w-full")
            ui.checkbox("Recursive", value=True).bind_value(self, "recursive")
            ui.checkbox("Skip images smaller than ROI").bind_value(self, "skip_oversize")

            ui.button("Crop all", on_click=self._run).props("color=primary")
            self.status = ui.label("").classes("text-sm")

        self.recursive = True
        self.skip_oversize = False

    def _on_upload(self, e: events.UploadEventArguments) -> None:
        data = e.content.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            ui.notify("Could not decode image", type="negative")
            return
        self.example = img
        ok, png = cv2.imencode(".png", img)
        if not ok:
            ui.notify("Could not re-encode image", type="negative")
            return
        import base64
        self.example_data_url = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()
        self.image.set_source(self.example_data_url)
        self._reset()

    def _on_click(self, e: events.MouseEventArguments) -> None:
        if self.example is None:
            return
        x, y = int(e.image_x), int(e.image_y)
        h, w = self.example.shape[:2]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        self.corners.append((x, y))
        if len(self.corners) > 2:
            self.corners = [self.corners[-1]]
        self._refresh_overlay()

    def _reset(self) -> None:
        self.corners = []
        self.roi = None
        self.image.content = ""
        self.roi_label.text = "ROI: (click two corners on the example)"

    def _refresh_overlay(self) -> None:
        if not self.corners:
            self.image.content = ""
            return
        marks = "".join(
            f'<circle cx="{x}" cy="{y}" r="6" fill="none" stroke="red" stroke-width="2"/>'
            for x, y in self.corners
        )
        rect = ""
        if len(self.corners) == 2:
            (x1, y1), (x2, y2) = self.corners
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            if w > 0 and h > 0:
                self.roi = Roi(x, y, w, h)
                self.roi_label.text = f"ROI: x={x} y={y} w={w} h={h}"
                rect = (
                    f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                    'fill="none" stroke="lime" stroke-width="3"/>'
                )
        self.image.content = marks + rect

    def _run(self) -> None:
        if self.roi is None:
            ui.notify("Set the ROI first (two clicks on the example).", type="warning")
            return
        src = Path(self.input_dir.value or "").expanduser()
        dst = Path(self.output_dir.value or "").expanduser()
        if not src.is_dir():
            ui.notify(f"Input is not a directory: {src}", type="negative")
            return
        try:
            written, skipped = crop_directory(
                src, dst, self.roi,
                recursive=self.recursive,
                skip_oversize=self.skip_oversize,
            )
        except (ValueError, NotADirectoryError) as exc:
            ui.notify(str(exc), type="negative")
            return
        self.status.text = f"Done: {written} written, {skipped} skipped -> {dst}"
        ui.notify(self.status.text, type="positive")


CropApp()
ui.run(title="ROI Cropper", reload=False)
