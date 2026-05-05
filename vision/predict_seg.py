"""predict_seg.py - run a trained checkpoint over an image (or directory)
and save predicted class-id masks. Optionally writes a colorized overlay.

    python vision/predict_seg.py \\
        --checkpoint runs/exp1/best.pt \\
        --input newdata/ \\
        --out preds/ \\
        --save-overlay
"""
from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True, help="single image or directory")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--save-overlay", action="store_true")
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
            if args.save_overlay:
                cv2.imwrite(str(args.out / f"{f.stem}_overlay.jpg"), overlay(bgr, pred))
    print(f"wrote {len(files)} predictions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
