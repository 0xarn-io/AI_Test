"""Tests for vision.crop_to_roi — exercises the pure logic without a GUI."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from vision.crop_to_roi import Roi, crop, crop_directory, iter_images, main


def test_roi_parse_comma():
    assert Roi.parse("10,20,30,40").as_tuple() == (10, 20, 30, 40)


def test_roi_parse_whitespace():
    assert Roi.parse("10 20 30 40").as_tuple() == (10, 20, 30, 40)


def test_roi_parse_rejects_wrong_arity():
    with pytest.raises(ValueError):
        Roi.parse("10,20,30")


def test_roi_rejects_zero_size():
    with pytest.raises(ValueError):
        Roi(0, 0, 0, 10)


def test_roi_rejects_negative_origin():
    with pytest.raises(ValueError):
        Roi(-1, 0, 10, 10)


def test_crop_returns_correct_subimage():
    img = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    out = crop(img, Roi(10, 20, 30, 40))
    assert out.shape == (40, 30, 3)
    assert np.array_equal(out, img[20:60, 10:40])


def test_crop_rejects_out_of_bounds():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        crop(img, Roi(40, 40, 20, 20))


def _write_image(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (100, 100)) -> None:
    img = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), img)


def test_iter_images_sorted_and_filtered(tmp_path: Path):
    _write_image(tmp_path / "b.jpg", (0, 0, 0))
    _write_image(tmp_path / "a.png", (0, 0, 0))
    (tmp_path / "notes.txt").write_text("ignore me")
    found = iter_images(tmp_path, recursive=False)
    assert [p.name for p in found] == ["a.png", "b.jpg"]


def test_crop_directory_preserves_structure(tmp_path: Path):
    src = tmp_path / "in"
    dst = tmp_path / "out"
    _write_image(src / "top.jpg", (10, 20, 30))
    _write_image(src / "sub" / "nested.jpg", (40, 50, 60))

    written, skipped = crop_directory(src, dst, Roi(0, 0, 50, 50))
    assert (written, skipped) == (2, 0)
    assert (dst / "top.jpg").exists()
    assert (dst / "sub" / "nested.jpg").exists()
    out = cv2.imread(str(dst / "top.jpg"))
    assert out.shape == (50, 50, 3)


def test_crop_directory_skip_oversize(tmp_path: Path):
    src = tmp_path / "in"
    dst = tmp_path / "out"
    _write_image(src / "small.jpg", (0, 0, 0), size=(20, 20))
    _write_image(src / "ok.jpg", (0, 0, 0), size=(100, 100))

    written, skipped = crop_directory(src, dst, Roi(0, 0, 50, 50), skip_oversize=True)
    assert (written, skipped) == (1, 1)
    assert (dst / "ok.jpg").exists()
    assert not (dst / "small.jpg").exists()


def test_crop_directory_errors_on_oversize_by_default(tmp_path: Path):
    src = tmp_path / "in"
    dst = tmp_path / "out"
    _write_image(src / "small.jpg", (0, 0, 0), size=(20, 20))
    with pytest.raises(ValueError):
        crop_directory(src, dst, Roi(0, 0, 50, 50))


def test_crop_directory_renames_sequentially(tmp_path: Path):
    src = tmp_path / "in"
    dst = tmp_path / "out"
    _write_image(src / "zzz.jpg", (0, 0, 0))
    _write_image(src / "aaa.jpg", (0, 0, 0))

    written, skipped = crop_directory(
        src, dst, Roi(0, 0, 50, 50), rename=("img_", 4, 1)
    )
    assert (written, skipped) == (2, 0)
    assert sorted(p.name for p in dst.iterdir()) == ["img_0001.jpg", "img_0002.jpg"]


def test_main_cli_with_explicit_roi(tmp_path: Path):
    src = tmp_path / "in"
    dst = tmp_path / "out"
    roi_file = tmp_path / "roi.txt"
    _write_image(src / "a.jpg", (0, 0, 0))

    rc = main([
        "--input", str(src),
        "--output", str(dst),
        "--roi", "0,0,50,50",
        "--roi-out", str(roi_file),
    ])
    assert rc == 0
    assert (dst / "a.jpg").exists()
    assert roi_file.read_text().strip() == "0,0,50,50"
