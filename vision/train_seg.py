"""train_seg.py - train a U-Net on a Label Studio COCO segmentation export.

The default layout is::

    AI_Test/
        dataset/                  <-- unzip the Label Studio export here
            result.json
            images/
                0001.jpg ...
        runs/                     <-- training outputs land here

With that layout, just run::

    python -m vision.train_seg

Override paths if your layout differs::

    python -m vision.train_seg --data /elsewhere/my_export

Saves the best (by foreground mIoU) checkpoint to ``<out>/best.pt``.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from vision.dataset import CocoSegDataset

log = logging.getLogger(__name__)


def per_class_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> list[float]:
    ious: list[float] = []
    pf = pred.reshape(-1)
    tf = target.reshape(-1)
    for c in range(num_classes):
        p = pf == c
        t = tf == c
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        ious.append(inter / union if union > 0 else float("nan"))
    return ious


def resolve_paths(data: Path, coco: Path | None, images: Path | None) -> tuple[Path, Path]:
    """Pick the COCO JSON and images dir for a dataset folder."""
    if coco is None:
        candidates = sorted(data.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No .json found in {data}. Unzip the COCO export there, "
                "or pass --coco explicitly."
            )
        # prefer 'result.json' if present; else first match
        coco = next((c for c in candidates if c.name == "result.json"), candidates[0])
    if images is None:
        first = json.loads(coco.read_text())["images"][0]["file_name"]
        if "/" in first or "\\" in first:
            images = coco.parent
        elif (coco.parent / "images").is_dir():
            images = coco.parent / "images"
        else:
            images = coco.parent
    return coco, images


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=Path("dataset"),
                   help="Folder containing result.json + images/ (default: ./dataset)")
    p.add_argument("--coco", type=Path, help="Override: path to the COCO JSON")
    p.add_argument("--images", type=Path, help="Override: directory the JSON file_names are relative to")
    p.add_argument("--out", type=Path, default=Path("runs/exp1"),
                   help="Run directory (default: ./runs/exp1)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.coco, args.images = resolve_paths(args.data, args.coco, args.images)
    log.info("coco=%s images=%s out=%s", args.coco, args.images, args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    # split image ids by random shuffle
    full = CocoSegDataset(args.coco, args.images)
    ids = list(full.ids)
    random.shuffle(ids)
    n_val = max(1, int(round(len(ids) * args.val_split)))
    val_ids, train_ids = ids[:n_val], ids[n_val:]
    log.info("train=%d val=%d classes=%s", len(train_ids), len(val_ids), full.class_names)

    train_ds = CocoSegDataset(args.coco, args.images, train_ids, augment=True)
    val_ds = CocoSegDataset(args.coco, args.images, val_ids, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
    )

    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=full.num_classes,
    ).to(args.device)

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = smp.losses.DiceLoss(mode="multiclass")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_miou = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
            logits = model(x)
            loss = ce_loss(logits, y) + dice_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= max(1, len(train_ds))

        # validation
        model.eval()
        per_class = [[] for _ in range(full.num_classes)]
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(args.device), y.to(args.device)
                pred = model(x).argmax(1)
                for ip, iy in zip(pred, y):
                    for c, v in enumerate(per_class_iou(ip, iy, full.num_classes)):
                        if v == v:  # not NaN
                            per_class[c].append(v)
        means = [sum(c) / len(c) if c else 0.0 for c in per_class]
        miou_fg = sum(means[1:]) / max(1, len(means) - 1)
        log.info(
            "epoch %d/%d  loss=%.4f  mIoU(fg)=%.4f  per-class=%s",
            epoch, args.epochs, train_loss, miou_fg,
            ", ".join(f"{n}={v:.3f}" for n, v in zip(full.class_names, means)),
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "miou_fg": miou_fg, "per_class": means})

        if miou_fg > best_miou:
            best_miou = miou_fg
            torch.save({
                "model_state": model.state_dict(),
                "encoder": args.encoder,
                "num_classes": full.num_classes,
                "class_names": full.class_names,
                "epoch": epoch,
                "miou_fg": miou_fg,
            }, args.out / "best.pt")
            log.info("saved best.pt (mIoU fg = %.4f)", miou_fg)

    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    log.info("done. best val mIoU(fg) = %.4f -> %s/best.pt", best_miou, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
