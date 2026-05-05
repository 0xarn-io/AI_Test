"""crop_ui.py — NiceGUI front-end for the ROI cropper.

Usage:
    python vision/crop_ui.py
    # then open http://localhost:8190

Workflow:
    1. Upload an example image.
    2. (Optional) Set target W x H and tick "Lock to target size" — the
       ROI will be exactly that many pixels, centered on your last click.
    3. Click two opposite corners (or one, if locked) to set the ROI.
       Use the arrow buttons to nudge it pixel-perfect.
    4. The preview shows the exact pixels every output image will get.
    5. Pick input/output directories and press "Crop all".
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

import cv2
import numpy as np
from nicegui import events, ui

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision.crop_to_roi import Roi, crop, crop_directory


def _png_data_url(img: np.ndarray) -> str:
    ok, png = cv2.imencode(".png", img)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()


class CropApp:
    def __init__(self) -> None:
        self.example: np.ndarray | None = None
        self.corners: list[tuple[int, int]] = []
        self.roi: Roi | None = None
        self.recursive = True
        self.skip_oversize = False

        with ui.column().classes("w-full max-w-4xl mx-auto gap-4 p-4"):
            ui.label("ROI Cropper").classes("text-2xl font-bold")
            ui.label(
                "Upload an example, set a target size if you have one, click "
                "to place the ROI, and use the arrows to fine-tune."
            ).classes("text-sm text-gray-600")

            ui.upload(
                label="Example image",
                auto_upload=True,
                on_upload=self._on_upload,
            ).props("accept=image/*").classes("w-full")

            with ui.row().classes("w-full items-end gap-3 flex-wrap"):
                self.target_w = ui.number("Target W", value=864, format="%d", min=1).classes("w-28")
                self.target_h = ui.number("Target H", value=832, format="%d", min=1).classes("w-28")
                self.lock_size = ui.checkbox("Lock to target size", value=True)
                ui.button("Snap to /32", on_click=self._snap_to_32).props("flat").tooltip(
                    "Round target W/H down to nearest multiple of 32 (CNN-friendly)"
                )

            with ui.row().classes("w-full gap-6 flex-wrap"):
                with ui.column().classes("flex-grow min-w-[400px]"):
                    ui.label("Example (click to place ROI)").classes("text-sm font-semibold")
                    self.image = ui.interactive_image(
                        "",
                        on_mouse=self._on_click,
                        events=["mousedown"],
                        cross=True,
                    ).classes("w-full")
                with ui.column().classes("min-w-[280px]"):
                    ui.label("Crop preview (1:1 pixels)").classes("text-sm font-semibold")
                    self.preview = ui.image("").classes("border border-gray-300")
                    self.roi_label = ui.label("ROI: (click on the example)").classes("text-xs font-mono")

            with ui.row().classes("items-center gap-2"):
                ui.label("Nudge:")
                self.step = ui.number("step px", value=1, format="%d", min=1).classes("w-24")
                ui.button(icon="arrow_back", on_click=lambda: self._nudge(-1, 0)).props("dense round")
                ui.button(icon="arrow_upward", on_click=lambda: self._nudge(0, -1)).props("dense round")
                ui.button(icon="arrow_downward", on_click=lambda: self._nudge(0, 1)).props("dense round")
                ui.button(icon="arrow_forward", on_click=lambda: self._nudge(1, 0)).props("dense round")
                ui.button("Reset", on_click=self._reset).props("flat")

            with ui.row().classes("w-full items-end gap-2"):
                self.input_dir = ui.input("Input directory").classes("flex-grow")
                ui.button("Browse", on_click=lambda: self._pick_folder(self.input_dir)).props("flat")
            with ui.row().classes("w-full items-end gap-2"):
                self.output_dir = ui.input("Output directory").classes("flex-grow")
                ui.button(
                    "Browse",
                    on_click=lambda: self._pick_folder(self.output_dir, allow_create=True),
                ).props("flat")
            ui.checkbox("Recursive", value=True).bind_value(self, "recursive")
            ui.checkbox("Skip images smaller than ROI").bind_value(self, "skip_oversize")

            ui.button("Crop all", on_click=self._run).props("color=primary")
            self.status = ui.label("").classes("text-sm")

    # --- example image ---------------------------------------------------

    def _on_upload(self, e: events.UploadEventArguments) -> None:
        data = e.content.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            ui.notify("Could not decode image", type="negative")
            return
        self.example = img
        self.image.set_source(_png_data_url(img))
        self._reset()

    # --- ROI placement ---------------------------------------------------

    def _on_click(self, e: events.MouseEventArguments) -> None:
        if self.example is None:
            return
        x, y = int(e.image_x), int(e.image_y)
        h, w = self.example.shape[:2]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        self.corners.append((x, y))
        if self.lock_size.value:
            self.corners = [self.corners[-1]]
        elif len(self.corners) > 2:
            self.corners = [self.corners[-1]]
        self._recompute()

    def _snap_to_32(self) -> None:
        self.target_w.value = max(32, int(self.target_w.value) // 32 * 32)
        self.target_h.value = max(32, int(self.target_h.value) // 32 * 32)
        self._recompute()

    def _nudge(self, dx: int, dy: int) -> None:
        if self.roi is None or self.example is None:
            return
        step = max(1, int(self.step.value))
        img_h, img_w = self.example.shape[:2]
        nx = max(0, min(self.roi.x + dx * step, img_w - self.roi.w))
        ny = max(0, min(self.roi.y + dy * step, img_h - self.roi.h))
        self.roi = Roi(nx, ny, self.roi.w, self.roi.h)
        if self.lock_size.value and self.corners:
            self.corners = [(nx + self.roi.w // 2, ny + self.roi.h // 2)]
        self._render()

    def _reset(self) -> None:
        self.corners = []
        self.roi = None
        self._render()

    def _recompute(self) -> None:
        self.roi = self._compute_roi()
        self._render()

    def _compute_roi(self) -> Roi | None:
        if self.example is None or not self.corners:
            return None
        img_h, img_w = self.example.shape[:2]
        if self.lock_size.value:
            tw, th = int(self.target_w.value), int(self.target_h.value)
            if tw > img_w or th > img_h:
                ui.notify(f"Target {tw}x{th} larger than image {img_w}x{img_h}", type="warning")
                return None
            cx, cy = self.corners[-1]
            x = max(0, min(cx - tw // 2, img_w - tw))
            y = max(0, min(cy - th // 2, img_h - th))
            return Roi(x, y, tw, th)
        if len(self.corners) == 2:
            (x1, y1), (x2, y2) = self.corners
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            if w > 0 and h > 0:
                return Roi(x, y, w, h)
        return None

    def _render(self) -> None:
        if self.roi is None:
            self.image.content = "".join(
                f'<circle cx="{x}" cy="{y}" r="6" fill="none" stroke="red" stroke-width="2"/>'
                for x, y in self.corners
            )
            self.preview.set_source("")
            self.roi_label.text = (
                "ROI: (click on the example)" if not self.corners
                else "ROI: need a second click (lock disabled)"
            )
            return
        r = self.roi
        self.image.content = (
            f'<rect x="{r.x}" y="{r.y}" width="{r.w}" height="{r.h}" '
            'fill="none" stroke="lime" stroke-width="3"/>'
        )
        self.roi_label.text = f"ROI: x={r.x} y={r.y} w={r.w} h={r.h}"
        if self.example is not None:
            self.preview.set_source(_png_data_url(crop(self.example, r)))

    # --- folder picker ---------------------------------------------------

    async def _pick_folder(self, target: ui.input, allow_create: bool = False) -> None:
        start = Path(target.value).expanduser() if target.value else Path.cwd()
        if not start.is_dir():
            start = Path.cwd()
        chosen = await FolderPicker(start, allow_create=allow_create)
        if chosen is not None:
            target.value = str(chosen)

    # --- run -------------------------------------------------------------

    def _run(self) -> None:
        if self.roi is None:
            ui.notify("Set the ROI first.", type="warning")
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
        import json as _json
        manifest = {
            "roi": {"x": self.roi.x, "y": self.roi.y, "w": self.roi.w, "h": self.roi.h},
            "source": str(src),
        }
        (dst / "roi.json").write_text(_json.dumps(manifest, indent=2))
        self.status.text = f"Done: {written} written, {skipped} skipped -> {dst}"
        ui.notify(self.status.text, type="positive")


class FolderPicker(ui.dialog):
    """Server-side folder picker. ``await FolderPicker(start)`` returns a Path or None."""

    def __init__(self, start: Path, allow_create: bool = False) -> None:
        super().__init__()
        self.path = start.resolve()
        self.allow_create = allow_create
        with self, ui.card().classes("w-[640px] max-w-[90vw]"):
            self.path_label = ui.label().classes("font-mono text-sm break-all")
            with ui.row().classes("gap-2"):
                ui.button("Up", on_click=self._go_up).props("flat dense")
                if allow_create:
                    ui.button("New folder", on_click=self._new_folder).props("flat dense")
            self.list_container = ui.column().classes("w-full max-h-80 overflow-auto gap-0")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=lambda: self.submit(None)).props("flat")
                ui.button(
                    "Select this folder",
                    on_click=lambda: self.submit(self.path),
                ).props("color=primary")
        self._refresh()

    def _refresh(self) -> None:
        self.path_label.text = str(self.path)
        self.list_container.clear()
        try:
            entries = sorted(p for p in self.path.iterdir() if p.is_dir() and not p.name.startswith("."))
        except PermissionError:
            entries = []
            with self.list_container:
                ui.label("(permission denied)").classes("text-xs text-red-500")
        with self.list_container:
            for entry in entries:
                ui.button(
                    f"📁  {entry.name}",
                    on_click=lambda _, p=entry: self._enter(p),
                ).props("flat align=left").classes("w-full justify-start")

    def _enter(self, p: Path) -> None:
        self.path = p
        self._refresh()

    def _go_up(self) -> None:
        if self.path.parent != self.path:
            self.path = self.path.parent
            self._refresh()

    async def _new_folder(self) -> None:
        name = await NamePrompt()
        if not name:
            return
        try:
            (self.path / name).mkdir(parents=False, exist_ok=False)
        except OSError as exc:
            ui.notify(f"Could not create: {exc}", type="negative")
            return
        self._refresh()


class NamePrompt(ui.dialog):
    """Single-line text prompt; ``await NamePrompt()`` returns str or None."""

    def __init__(self) -> None:
        super().__init__()
        with self, ui.card():
            field = ui.input("Folder name").classes("w-64")
            with ui.row().classes("justify-end gap-2"):
                ui.button("Cancel", on_click=lambda: self.submit(None)).props("flat")
                ui.button("Create", on_click=lambda: self.submit(field.value)).props("color=primary")


CropApp()
ui.run(title="ROI Cropper", reload=False, port=8190)
