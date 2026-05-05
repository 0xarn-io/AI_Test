"""remap_dataset.py - re-crop originals to a new ROI and translate any
existing Label Studio annotations from the old crop space to the new one.

Background
----------
You labeled images at one ROI (the "old" crop), and now want a different
ROI (the "new" crop). Both are taken from the same originals. Pixel
coordinates in old-crop space map to new-crop space by:

    x_new = x_old + (old_roi.x - new_roi.x)
    y_new = y_old + (old_roi.y - new_roi.y)

This script applies that shift to every supported annotation kind, drops
or clips items that fall outside the new ROI, and writes new images
alongside.

Supported inputs
----------------
* Originals directory: full-resolution frames the old crop came from.
* PNG mask exports (Label Studio "Brush labels to numpy and image"):
  one or more PNG masks per image, sized to the *old* crop. Padded to
  the original frame, then cropped at the new ROI.
* JSON / JSON-MIN export: rectanglelabels, polygonlabels, keypointlabels
  whose coordinates are in PERCENT of the old crop dimensions. Converted
  to pixels, shifted, clipped to new bounds, converted back to percent.
* brushlabels in JSON (RLE) are NOT supported here -- export as PNG masks
  instead.

Usage
-----
    python vision/remap_dataset.py \\
        --originals Data/ \\
        --old-roi 817,351,855,831 \\
        --new-roi 800,340,896,864 \\
        --out-images newdata/ \\
        [--masks masks_old/ --out-masks masks_new/] \\
        [--labels labels.json --out-labels labels_new.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision.crop_to_roi import Roi, crop, iter_images

log = logging.getLogger(__name__)


def crop_originals(originals: Path, new_roi: Roi, out: Path) -> int:
    """Crop every image under ``originals`` at ``new_roi`` into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in iter_images(originals, recursive=True):
        img = cv2.imread(str(src))
        if img is None:
            log.warning("unreadable: %s", src)
            continue
        h, w = img.shape[:2]
        if new_roi.x + new_roi.w > w or new_roi.y + new_roi.h > h:
            log.warning("%s: ROI exceeds image %dx%d - skipping", src.name, w, h)
            continue
        rel = src.relative_to(originals)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), crop(img, new_roi))
        n += 1
    return n


def remap_mask(old_mask: np.ndarray, original_shape: tuple[int, int],
               old_roi: Roi, new_roi: Roi) -> np.ndarray:
    """Place an old-crop-sized mask back onto the original frame, then crop at new_roi."""
    H, W = original_shape
    if old_mask.shape[:2] != (old_roi.h, old_roi.w):
        raise ValueError(
            f"mask shape {old_mask.shape[:2]} != old ROI {(old_roi.h, old_roi.w)}"
        )
    canvas = np.zeros((H, W) + old_mask.shape[2:], dtype=old_mask.dtype)
    canvas[old_roi.y:old_roi.y + old_roi.h, old_roi.x:old_roi.x + old_roi.w] = old_mask
    return canvas[new_roi.y:new_roi.y + new_roi.h, new_roi.x:new_roi.x + new_roi.w]


def remap_masks_dir(masks: Path, out: Path, original_shape: tuple[int, int],
                    old_roi: Roi, new_roi: Roi) -> int:
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in iter_images(masks, recursive=True, extensions=(".png",)):
        m = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if m is None:
            log.warning("unreadable mask: %s", src)
            continue
        try:
            new = remap_mask(m, original_shape, old_roi, new_roi)
        except ValueError as e:
            log.warning("%s: %s", src.name, e)
            continue
        rel = src.relative_to(masks)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), new)
        n += 1
    return n


def _shift_point_pct(x_pct: float, y_pct: float, old_roi: Roi, new_roi: Roi) -> tuple[float, float]:
    """Translate a (x%, y%) Label Studio point from old-crop space to new-crop space."""
    x_px = x_pct / 100.0 * old_roi.w
    y_px = y_pct / 100.0 * old_roi.h
    nx = x_px + (old_roi.x - new_roi.x)
    ny = y_px + (old_roi.y - new_roi.y)
    return nx / new_roi.w * 100.0, ny / new_roi.h * 100.0


def _clip_pct(v: float) -> float:
    return max(0.0, min(100.0, v))


def remap_label_result(result: dict, old_roi: Roi, new_roi: Roi) -> dict | None:
    """Return a remapped Label Studio result dict, or None if it falls outside."""
    r = deepcopy(result)
    r["original_width"] = new_roi.w
    r["original_height"] = new_roi.h
    val = r.get("value", {})
    rtype = r.get("type")

    if rtype == "rectanglelabels":
        x, y = _shift_point_pct(val["x"], val["y"], old_roi, new_roi)
        # bottom-right
        x2, y2 = _shift_point_pct(val["x"] + val["width"], val["y"] + val["height"], old_roi, new_roi)
        x_c, y_c = _clip_pct(x), _clip_pct(y)
        x2_c, y2_c = _clip_pct(x2), _clip_pct(y2)
        if x2_c <= x_c or y2_c <= y_c:
            return None
        val["x"], val["y"] = x_c, y_c
        val["width"], val["height"] = x2_c - x_c, y2_c - y_c
        return r

    if rtype == "polygonlabels":
        pts = [_shift_point_pct(px, py, old_roi, new_roi) for px, py in val["points"]]
        clipped = [(_clip_pct(px), _clip_pct(py)) for px, py in pts]
        # drop if all points hit the same edge (degenerate after clipping)
        xs = {round(px, 3) for px, _ in clipped}
        ys = {round(py, 3) for _, py in clipped}
        if len(xs) <= 1 or len(ys) <= 1:
            return None
        val["points"] = [list(p) for p in clipped]
        return r

    if rtype == "keypointlabels":
        x, y = _shift_point_pct(val["x"], val["y"], old_roi, new_roi)
        if not (0 <= x <= 100 and 0 <= y <= 100):
            return None
        val["x"], val["y"] = x, y
        return r

    log.warning("unsupported result type %r - leaving as-is", rtype)
    return r


def remap_labels_json(in_path: Path, out_path: Path, old_roi: Roi, new_roi: Roi) -> tuple[int, int]:
    """Remap a Label Studio JSON / JSON-MIN file. Returns (kept, dropped)."""
    data = json.loads(in_path.read_text())
    if not isinstance(data, list):
        raise ValueError("Expected a list of tasks at the top level (JSON / JSON-MIN export).")
    kept = dropped = 0
    for task in data:
        for ann in task.get("annotations", []):
            new_results = []
            for res in ann.get("result", []):
                remapped = remap_label_result(res, old_roi, new_roi)
                if remapped is None:
                    dropped += 1
                else:
                    new_results.append(remapped)
                    kept += 1
            ann["result"] = new_results
    out_path.write_text(json.dumps(data, indent=2))
    return kept, dropped


def _parse_roi(spec: str) -> Roi:
    return Roi.parse(spec)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--originals", type=Path, required=True)
    p.add_argument("--old-roi", type=_parse_roi, required=True, help="x,y,w,h of the existing crop in the originals")
    p.add_argument("--new-roi", type=_parse_roi, required=True, help="x,y,w,h of the new crop in the originals")
    p.add_argument("--out-images", type=Path, required=True)
    p.add_argument("--masks", type=Path, help="dir of PNG masks at old-crop size")
    p.add_argument("--out-masks", type=Path)
    p.add_argument("--labels", type=Path, help="Label Studio JSON / JSON-MIN export")
    p.add_argument("--out-labels", type=Path)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")

    n = crop_originals(args.originals, args.new_roi, args.out_images)
    log.info("cropped %d originals -> %s", n, args.out_images)

    if args.masks:
        if not args.out_masks:
            p.error("--out-masks required with --masks")
        # peek first original to get full-frame dims
        first = next(iter(iter_images(args.originals, recursive=True)))
        H, W = cv2.imread(str(first)).shape[:2]
        n = remap_masks_dir(args.masks, args.out_masks, (H, W), args.old_roi, args.new_roi)
        log.info("remapped %d masks -> %s", n, args.out_masks)

    if args.labels:
        if not args.out_labels:
            p.error("--out-labels required with --labels")
        kept, dropped = remap_labels_json(args.labels, args.out_labels, args.old_roi, args.new_roi)
        log.info("labels: %d kept, %d dropped (fully outside new ROI) -> %s",
                 kept, dropped, args.out_labels)

    return 0


if __name__ == "__main__":
    sys.exit(main())
