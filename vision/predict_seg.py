"""predict_seg.py - run a trained checkpoint over an image (or directory)
and save predicted class-id masks. Optionally writes a colorized overlay
and/or a JSON file with per-class presence, instance counts, total
pixel area, and bounding boxes.

    python vision/predict_seg.py \\
        --checkpoint runs/exp1/best.pt \\
        --input newdata/ \\
        --out preds/ \\
        --save-overlay --analyze
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

import segmentation_models_pytorch as smp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision.crop_to_roi import iter_images

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

_PALETTE = np.array([
    [0, 0, 0],
    [220, 50, 50], [50, 220, 50], [50, 50, 220],
    [220, 220, 50], [220, 50, 220], [50, 220, 220],
    [180, 120, 60], [120, 60, 180],
], dtype=np.uint8)


def overlay(bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    color = _PALETTE[mask % len(_PALETTE)]
    return cv2.addWeighted(bgr, 1 - alpha, color[..., ::-1], alpha, 0)


def analyze(mask: np.ndarray, class_names: list[str], min_area: int = 100) -> dict:
    """Count instances and locate each per non-background class."""
    out: dict = {"image_size": list(mask.shape[:2]), "classes": {}}
    for cls_id, name in enumerate(class_names):
        if cls_id == 0:
            continue
        binary = (mask == cls_id).astype(np.uint8)
        n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        instances = []
        for i in range(1, n_lab):
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue
            instances.append({"bbox": [int(x), int(y), int(w), int(h)], "area_px": int(area)})
        instances.sort(key=lambda d: d["area_px"], reverse=True)
        total = int(binary.sum())
        out["classes"][name] = {
            "present": bool(instances),
            "instances": len(instances),
            "total_area_px": total,
            "coverage": float(total) / mask.size,
            "boxes": instances,
        }
    return out


def annotate(bgr: np.ndarray, analysis: dict, class_names: list[str]) -> np.ndarray:
    """Draw bounding boxes and class labels onto the overlay."""
    out = bgr.copy()
    for cls_id, name in enumerate(class_names):
        if cls_id == 0 or name not in analysis["classes"]:
            continue
        color_rgb = _PALETTE[cls_id % len(_PALETTE)]
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        for inst in analysis["classes"][name]["boxes"]:
            x, y, w, h = inst["bbox"]
            cv2.rectangle(out, (x, y), (x + w, y + h), color_bgr, 2)
            label = f"{name} {inst['area_px']}px"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x, y - th - 4), (x + tw + 4, y), color_bgr, -1)
            cv2.putText(out, label, (x + 2, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True, help="single image or directory")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--save-overlay", action="store_true")
    p.add_argument("--analyze", action="store_true",
                   help="Write per-image JSON with presence/counts/bboxes per class")
    p.add_argument("--min-area", type=int, default=100,
                   help="Drop instances smaller than this many pixels (default 100)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    encoder = ck.get("encoder", "resnet34")
    num_classes = ck["num_classes"]
    class_names = ck.get("class_names", [str(i) for i in range(num_classes)])

    model = smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=num_classes)
    model.load_state_dict(ck["model_state"])
    model.to(args.device).eval()

    args.out.mkdir(parents=True, exist_ok=True)
    files = [args.input] if args.input.is_file() else iter_images(args.input, recursive=False)
    if not files:
        print(f"no images found in {args.input}", file=sys.stderr)
        return 1

    print(f"classes: {class_names}")
    summary = []
    with torch.no_grad():
        for f in files:
            bgr = cv2.imread(str(f))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            x = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
            t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(args.device)
            pred = model(t).argmax(1)[0].cpu().numpy().astype(np.uint8)
            cv2.imwrite(str(args.out / f"{f.stem}_mask.png"), pred)

            ana = analyze(pred, class_names, min_area=args.min_area) if args.analyze else None
            if ana is not None:
                ana["image"] = f.name
                (args.out / f"{f.stem}.json").write_text(json.dumps(ana, indent=2))
                summary.append({
                    "image": f.name,
                    "counts": {k: v["instances"] for k, v in ana["classes"].items()},
                })

            if args.save_overlay:
                ov = overlay(bgr, pred)
                if ana is not None:
                    ov = annotate(ov, ana, class_names)
                cv2.imwrite(str(args.out / f"{f.stem}_overlay.jpg"), ov)

    if summary:
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {len(files)} predictions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
