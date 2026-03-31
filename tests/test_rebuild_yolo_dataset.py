"""Tests for rebuilding YOLO dataset from Label Studio data."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np

from scripts import rebuild_yolo_dataset_from_label_studio as rebuild


def _make_points(offset: float) -> list[dict]:
    result = []
    for row in range(4):
        for col in range(4):
            result.append(
                {
                    "from_name": "corners",
                    "to_name": "image",
                    "type": "keypointlabels",
                    "value": {
                        "x": offset + col * 10.0,
                        "y": offset + row * 10.0,
                        "width": 0.3,
                        "keypointlabels": [f"P{row}{col}"],
                    },
                }
            )
    return result


def _init_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        create table project (id integer primary key, model_version text);
        create table task (
          id integer primary key,
          project_id integer,
          data text
        );
        create table prediction (
          id integer primary key,
          task_id integer,
          model_version text,
          result text,
          created_at text
        );
        create table task_completion (
          id integer primary key,
          task_id integer,
          result text,
          updated_at text
        );
        """
    )
    conn.execute("insert into project(id, model_version) values(2, '20260331.1')")
    conn.execute(
        "insert into task(id, project_id, data) values(1, 2, ?)",
        (json.dumps({"image": "/data/upload/2/abc123-IMG_0001.jpg"}),),
    )
    conn.execute(
        "insert into task(id, project_id, data) values(2, 2, ?)",
        (json.dumps({"image": "/data/upload/2/def456-IMG_0002.jpg"}),),
    )
    conn.execute(
        "insert into prediction(id, task_id, model_version, result, created_at) values(10, 1, '20260331.1', ?, '2026-03-31 08:00:00')",
        (json.dumps(_make_points(1.0)),),
    )
    conn.execute(
        "insert into prediction(id, task_id, model_version, result, created_at) values(11, 2, '20260331.1', ?, '2026-03-31 08:00:00')",
        (json.dumps(_make_points(2.0)),),
    )
    conn.execute(
        "insert into task_completion(id, task_id, result, updated_at) values(20, 1, ?, '2026-03-31 09:00:00')",
        (json.dumps(_make_points(5.0)),),
    )
    conn.commit()
    conn.close()


def test_export_project_uses_annotations_only(tmp_path):
    db_path = tmp_path / "label_studio.sqlite3"
    _init_db(db_path)

    raw_images = tmp_path / "raw"
    raw_images.mkdir()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert cv2.imwrite(str(raw_images / "IMG_0001.jpg"), image)
    assert cv2.imwrite(str(raw_images / "IMG_0002.jpg"), image)

    dst_root = tmp_path / "yolo_grid_rebuild"
    stats = rebuild.export_project_to_yolo_dataset(
        db_path=db_path,
        project_id=2,
        raw_images_dir=raw_images,
        dst_root=dst_root,
    )

    assert stats == {
        "tasks_total": 2,
        "exported_total": 1,
        "exported_from_annotations": 1,
        "exported_from_predictions": 0,
        "skipped_missing_image": 0,
        "skipped_missing_labels": 1,
    }
    assert (dst_root / "images" / "train" / "IMG_0001.jpg").exists()
    assert not (dst_root / "images" / "train" / "IMG_0002.jpg").exists()

    label_1 = (dst_root / "labels" / "train" / "IMG_0001.txt").read_text(encoding="utf-8")
    yaml_text = (dst_root / "sudoku_grid.yaml").read_text(encoding="utf-8")

    assert "0.050000 0.050000" in label_1
    assert "train: images/train" in yaml_text
    assert "val: images/val" in yaml_text
