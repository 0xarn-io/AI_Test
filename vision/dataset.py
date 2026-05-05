"""dataset.py - turn a Label Studio COCO export into a torch semantic
segmentation dataset (polygons rasterized to a class-id mask)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from pycocotools.coco import COCO
except ImportError as e:
    raise ImportError(
        "pycocotools is required: pip install pycocotools"
    ) from e


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


class CocoSegDataset(Dataset):
    """One sample = (image_tensor [3,H,W] normalized, mask_tensor [H,W] long).

    Mask values: 0 = background, 1..N = the COCO categories sorted by id.
    """

    def __init__(
        self,
        coco_json: Path,
        images_dir: Path,
        image_ids: list[int] | None = None,
        augment: bool = False,
    ) -> None:
        self.coco = COCO(str(coco_json))
        self.images_dir = Path(images_dir)
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_to_class = {c: i + 1 for i, c in enumerate(self.cat_ids)}
        self.num_classes = len(self.cat_ids) + 1
        self.ids = list(image_ids) if image_ids is not None else sorted(self.coco.getImgIds())
        self.augment = augment

    @property
    def class_names(self) -> list[str]:
        names = ["background"]
        for c in self.cat_ids:
            names.append(self.coco.loadCats([c])[0]["name"])
        return names

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_id = self.ids[idx]
        info = self.coco.loadImgs([img_id])[0]
        img_path = self.images_dir / info["file_name"]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        mask = np.zeros((H, W), dtype=np.uint8)
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id])):
            cls = self.cat_to_class[ann["category_id"]]
            for poly in ann.get("segmentation", []):
                if not isinstance(poly, list) or len(poly) < 6:
                    continue
                pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], cls)

        if self.augment and np.random.rand() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        x = (rgb.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        img_t = torch.from_numpy(x.transpose(2, 0, 1)).contiguous()
        mask_t = torch.from_numpy(mask).long()
        return img_t, mask_t
