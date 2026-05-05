"""crop_to_roi.py — pick a region of interest on one example image, then
crop every image in a directory to the same box.

Two modes:

    # interactive: opens an OpenCV window on the example, drag a box, ENTER to confirm
    python vision/crop_to_roi.py --example sample.jpg --input raw/ --output cropped/

    # headless / repeatable: pass the box explicitly
    python vision/crop_to_roi.py --roi 120,80,640,480 --input raw/ --output cropped/

The interactive picker needs a display. The project's pinned wheel is
``opencv-python-headless``, which has no GUI — install ``opencv-python``
on the workstation where you do the picking, or supply ``--roi`` directly.

After an interactive pick the chosen box is printed so you can paste it
back as ``--roi`` next time, and (if ``--roi-out`` is given) written to
a small text file alongside the crops.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class Roi:
    x: int
    y: int
    w: int
    h: int

    def __post_init__(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise ValueError(f"ROI must have positive size, got w={self.w} h={self.h}")
        if self.x < 0 or self.y < 0:
            raise ValueError(f"ROI origin must be non-negative, got x={self.x} y={self.y}")

    @classmethod
    def parse(cls, spec: str) -> "Roi":
        """Parse 'x,y,w,h' (commas or whitespace)."""
        parts = spec.replace(",", " ").split()
        if len(parts) != 4:
            raise ValueError(f"ROI must be 'x,y,w,h', got {spec!r}")
        return cls(*(int(p) for p in parts))

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


def crop(img: np.ndarray, roi: Roi) -> np.ndarray:
    """Return the ROI sub-image. Raises if ROI extends past image bounds."""
    h, w = img.shape[:2]
    if roi.x + roi.w > w or roi.y + roi.h > h:
        raise ValueError(f"ROI {roi.as_tuple()} exceeds image {w}x{h}")
    return img[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]


def pick_roi_interactive(example_path: Path, window: str = "Select ROI - drag, then ENTER") -> Roi:
    """Open a window on the example image and let the user drag a box.

    Returns the chosen ROI. Raises RuntimeError if the OpenCV build has no
    GUI (e.g. opencv-python-headless) or the user cancels.
    """
    img = cv2.imread(str(example_path))
    if img is None:
        raise FileNotFoundError(f"Could not read example image: {example_path}")

    try:
        x, y, w, h = cv2.selectROI(window, img, showCrosshair=True, fromCenter=False)
    except cv2.error as e:
        raise RuntimeError(
            "Interactive ROI picker unavailable - this is opencv-python-headless. "
            "Install opencv-python for the GUI, or pass --roi x,y,w,h."
        ) from e
    finally:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass

    if w == 0 or h == 0:
        raise RuntimeError("ROI selection cancelled (empty box).")
    return Roi(int(x), int(y), int(w), int(h))


def iter_images(root: Path, recursive: bool, extensions: Iterable[str] = DEFAULT_EXTENSIONS) -> list[Path]:
    """List image files under root, sorted for stable output order."""
    exts = {e.lower() for e in extensions}
    walker = root.rglob("*") if recursive else root.iterdir()
    return sorted(p for p in walker if p.is_file() and p.suffix.lower() in exts)


def crop_directory(
    src_root: Path,
    dst_root: Path,
    roi: Roi,
    recursive: bool = True,
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
    skip_oversize: bool = False,
    rename: tuple[str, int, int] | None = None,
) -> tuple[int, int]:
    """Crop every image under src_root to roi and write to dst_root.

    By default preserves the source directory structure. If ``rename`` is
    given as ``(prefix, pad, start)``, output files are flat and named
    ``{prefix}{i:0{pad}d}{ext}`` where i counts up from ``start``.
    Returns ``(written, skipped)``.
    """
    if not src_root.is_dir():
        raise NotADirectoryError(f"input is not a directory: {src_root}")
    dst_root.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    for src in iter_images(src_root, recursive, extensions):
        img = cv2.imread(str(src))
        if img is None:
            log.warning("unreadable, skipping: %s", src)
            skipped += 1
            continue
        try:
            out = crop(img, roi)
        except ValueError as e:
            if skip_oversize:
                log.warning("%s: %s - skipping", src.name, e)
                skipped += 1
                continue
            raise

        if rename is not None:
            prefix, pad, start = rename
            dst = dst_root / f"{prefix}{written + start:0{pad}d}{src.suffix.lower()}"
        else:
            rel = src.relative_to(src_root)
            dst = dst_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dst), out):
            log.warning("failed to write: %s", dst)
            skipped += 1
            continue
        written += 1
    return written, skipped


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, required=True, help="directory of source images")
    p.add_argument("--output", type=Path, required=True, help="directory to write crops to")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--roi", type=Roi.parse, help="explicit ROI as x,y,w,h")
    src.add_argument("--example", type=Path, help="example image; opens GUI to pick ROI")
    p.add_argument("--no-recursive", action="store_true", help="do not descend into subdirs")
    p.add_argument("--skip-oversize", action="store_true",
                   help="skip images smaller than the ROI instead of erroring")
    p.add_argument("--roi-out", type=Path, help="write chosen ROI to this file")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")

    roi = args.roi if args.roi is not None else pick_roi_interactive(args.example)
    log.info("ROI: %d,%d,%d,%d", *roi.as_tuple())

    if args.roi_out:
        args.roi_out.write_text("{},{},{},{}\n".format(*roi.as_tuple()))

    written, skipped = crop_directory(
        args.input,
        args.output,
        roi,
        recursive=not args.no_recursive,
        skip_oversize=args.skip_oversize,
    )
    log.info("done: %d written, %d skipped", written, skipped)
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
