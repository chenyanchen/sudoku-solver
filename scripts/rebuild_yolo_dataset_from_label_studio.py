"""Rebuild raw YOLO dataset from Label Studio SQLite project data."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_backend.app import _extract_filename
from scripts.auto_label_yolo import keypoints_16_to_yolo_pose


def _write_dataset_yaml(dst_root: Path) -> None:
    (dst_root / "sudoku_grid.yaml").write_text(
        "\n".join(
            [
                f"path: {dst_root.resolve()}",
                "train: images/train",
                "val: images/val",
                "",
                "names:",
                "  0: sudoku_grid",
                "",
                "kpt_shape: [16, 2]",
                "flip_idx: []",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _annotation_to_label_line(result_json: str, width: int, height: int) -> str | None:
    try:
        result = json.loads(result_json)
    except json.JSONDecodeError:
        return None

    kpt_map: dict[str, tuple[float, float]] = {}
    seen_labels: list[str] = []
    for item in result:
        if item.get("type") != "keypointlabels":
            continue
        labels = item.get("value", {}).get("keypointlabels", [])
        if not labels:
            continue
        x_pct = item["value"]["x"]
        y_pct = item["value"]["y"]
        point = (x_pct / 100.0 * width, y_pct / 100.0 * height)
        label = labels[0]
        seen_labels.append(label)
        kpt_map[label] = point

    import numpy as np

    ordered = []
    duplicates = {label for label in seen_labels if seen_labels.count(label) > 1}
    for row in range(4):
        for col in range(4):
            label = f"P{row}{col}"
            if label not in kpt_map:
                return None
            ordered.append(kpt_map[label])
    if duplicates:
        return None
    keypoints = np.array(ordered, dtype=np.float32)
    return keypoints_16_to_yolo_pose(keypoints, width, height)


def _prediction_to_label_line(result_json: str, width: int, height: int) -> str | None:
    try:
        result = json.loads(result_json)
    except json.JSONDecodeError:
        return None

    kpt_map: dict[str, tuple[float, float]] = {}
    for item in result:
        if item.get("type") != "keypointlabels":
            continue
        labels = item.get("value", {}).get("keypointlabels", [])
        if not labels:
            continue
        x_pct = item["value"]["x"]
        y_pct = item["value"]["y"]
        kpt_map[labels[0]] = (x_pct / 100.0 * width, y_pct / 100.0 * height)

    ordered = []
    for row in range(4):
        for col in range(4):
            label = f"P{row}{col}"
            if label not in kpt_map:
                return None
            ordered.append(kpt_map[label])

    import numpy as np

    keypoints = np.array(ordered, dtype=np.float32)
    return keypoints_16_to_yolo_pose(keypoints, width, height)


def export_project_to_yolo_dataset(
    *,
    db_path: Path,
    project_id: int,
    raw_images_dir: Path,
    dst_root: Path,
) -> dict[str, int]:
    images_dir = dst_root / "images" / "train"
    labels_dir = dst_root / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    project = conn.execute("select model_version from project where id = ?", (project_id,)).fetchone()
    if project is None:
        raise RuntimeError(f"Project {project_id} not found in {db_path}")
    model_version = project["model_version"]

    tasks = conn.execute("select id, data from task where project_id = ? order by id", (project_id,)).fetchall()

    stats = {
        "tasks_total": len(tasks),
        "exported_total": 0,
        "exported_from_annotations": 0,
        "exported_from_predictions": 0,
        "skipped_missing_image": 0,
        "skipped_missing_labels": 0,
    }

    import cv2

    for task in tasks:
        task_data = {"data": json.loads(task["data"])}
        filename = _extract_filename(task_data)
        src_image = raw_images_dir / filename
        if not src_image.exists():
            stats["skipped_missing_image"] += 1
            continue

        image = cv2.imread(str(src_image))
        if image is None:
            stats["skipped_missing_image"] += 1
            continue
        height, width = image.shape[:2]

        annotation = conn.execute(
            "select result from task_completion where task_id = ? order by updated_at desc limit 1",
            (task["id"],),
        ).fetchone()
        label_line = None
        if annotation is not None:
            label_line = _annotation_to_label_line(annotation["result"], width, height)
            if label_line is not None:
                stats["exported_from_annotations"] += 1

        if label_line is None:
            prediction = conn.execute(
                """
                select result from prediction
                where task_id = ? and model_version = ?
                order by created_at desc limit 1
                """,
                (task["id"], model_version),
            ).fetchone()
            if prediction is None:
                prediction = conn.execute(
                    "select result from prediction where task_id = ? order by created_at desc limit 1",
                    (task["id"],),
                ).fetchone()
            if prediction is not None:
                label_line = _prediction_to_label_line(prediction["result"], width, height)
                if label_line is not None:
                    stats["exported_from_predictions"] += 1

        if label_line is None:
            stats["skipped_missing_labels"] += 1
            continue

        shutil.copy2(src_image, images_dir / src_image.name)
        (labels_dir / f"{src_image.stem}.txt").write_text(label_line + "\n", encoding="utf-8")
        stats["exported_total"] += 1

    conn.close()
    _write_dataset_yaml(dst_root)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild raw YOLO dataset from Label Studio SQLite")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path.home() / "data" / "label-studio" / "label_studio.sqlite3",
    )
    parser.add_argument("--project-id", type=int, default=2)
    parser.add_argument(
        "--raw-images-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "images",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=REPO_ROOT / "data" / "yolo_grid_rebuild",
    )
    args = parser.parse_args()

    stats = export_project_to_yolo_dataset(
        db_path=args.db_path,
        project_id=args.project_id,
        raw_images_dir=args.raw_images_dir,
        dst_root=args.dst_root,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
