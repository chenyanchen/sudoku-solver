"""Tests for YOLO training dataset preparation."""

from __future__ import annotations

from pathlib import Path

from scripts import train_yolo_grid


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_prepare_dataset_split_keeps_augmented_samples_out_of_val(tmp_path):
    dataset_root = tmp_path / "data" / "yolo_grid"
    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"
    labels_train.mkdir(parents=True, exist_ok=True)

    for stem in ("IMG_0001", "IMG_0002", "IMG_0003", "IMG_0004"):
        _touch(images_train / f"{stem}.jpg")
        (labels_train / f"{stem}.txt").write_text("label\n", encoding="utf-8")

    for stem in ("aug_0001", "aug_0002"):
        _touch(images_train / f"{stem}.jpg")
        (labels_train / f"{stem}.txt").write_text("label\n", encoding="utf-8")

    split_root = tmp_path / "split"
    yaml_path = train_yolo_grid.prepare_dataset_split(
        source_root=dataset_root,
        split_root=split_root,
        val_ratio=0.25,
        seed=7,
    )

    train_names = sorted(p.name for p in (split_root / "images" / "train").iterdir())
    val_names = sorted(p.name for p in (split_root / "images" / "val").iterdir())

    assert yaml_path == split_root / "sudoku_grid.yaml"
    assert val_names == ["IMG_0004.jpg"]
    assert "aug_0001.jpg" in train_names
    assert "aug_0002.jpg" in train_names
    assert all(not name.startswith("aug_") for name in val_names)
    assert sorted(name for name in train_names if not name.startswith("aug_")) == [
        "IMG_0001.jpg",
        "IMG_0002.jpg",
        "IMG_0003.jpg",
    ]


def test_parser_defaults_to_50_epochs():
    parser = train_yolo_grid.build_parser()

    args = parser.parse_args([])

    assert args.epochs == 50
