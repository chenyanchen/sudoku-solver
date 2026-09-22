from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from .schema import extract_filename, extract_textarea_binding, ordered_corner_labels


def fit_corners(
    annotation: dict[str, Any],
    task: dict[str, Any],
    parsed_label_config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    filename = extract_filename(task, parsed_label_config, repo_root)
    src_image = repo_root / "data" / "raw" / "images" / filename
    if not src_image.exists():
        return {"status": "skipped", "reason": f"source image not found: {filename}"}

    corner_map: dict[str, tuple[float, float]] = {}
    for item in annotation.get("result", []):
        if item.get("type") != "keypointlabels":
            continue
        labels = item.get("value", {}).get("keypointlabels", [])
        if not labels:
            continue
        corner_map[labels[0]] = (
            item["value"]["x"] / 100.0,
            item["value"]["y"] / 100.0,
        )

    expected = ordered_corner_labels(parsed_label_config)
    missing = [label for label in expected if label not in corner_map]
    if missing:
        return {"status": "skipped", "reason": f"missing corners: {missing}"}

    dst_dir = repo_root / "data" / "corner_cnn" / "images"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_img = dst_dir / Path(filename).name
    if not dst_img.exists():
        shutil.copy2(src_image, dst_img)

    csv_path = repo_root / "data" / "corner_cnn" / "labels.csv"
    fieldnames = [
        "image",
        "x_tl",
        "y_tl",
        "x_tr",
        "y_tr",
        "x_br",
        "y_br",
        "x_bl",
        "y_bl",
    ]
    row_data = {
        "image": Path(filename).name,
        "x_tl": f"{corner_map['TL'][0]:.6f}",
        "y_tl": f"{corner_map['TL'][1]:.6f}",
        "x_tr": f"{corner_map['TR'][0]:.6f}",
        "y_tr": f"{corner_map['TR'][1]:.6f}",
        "x_br": f"{corner_map['BR'][0]:.6f}",
        "y_br": f"{corner_map['BR'][1]:.6f}",
        "x_bl": f"{corner_map['BL'][0]:.6f}",
        "y_bl": f"{corner_map['BL'][1]:.6f}",
    }

    existing_rows = []
    found = False
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["image"] == Path(filename).name:
                    existing_rows.append(row_data)
                    found = True
                else:
                    existing_rows.append(row)

    if found:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
    else:
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row_data)

    return {"status": "ok", "saved": filename}


def fit_ocr(
    annotation: dict[str, Any],
    task: dict[str, Any],
    parsed_label_config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    filename = extract_filename(task, parsed_label_config, repo_root)
    from_name, _to_name = extract_textarea_binding(parsed_label_config)

    grid_lines: list[str] = []
    for item in annotation.get("result", []):
        if item.get("from_name") == from_name and item.get("type") == "textarea":
            grid_lines = item["value"].get("text", [])
            break

    if not grid_lines:
        return {"status": "skipped", "reason": f"no grid text for {filename}"}

    grid: list[list[int]] = []
    for line in grid_lines:
        row = [int(d) for d in line.strip().split()]
        if len(row) != 9:
            return {
                "status": "skipped",
                "reason": f"bad row length {len(row)} for {filename}",
            }
        grid.append(row)
    if len(grid) != 9:
        return {"status": "skipped", "reason": f"bad grid rows {len(grid)} for {filename}"}

    labels_path = repo_root / "data" / "labels" / "sudoku_labels.json"
    labels: dict[str, Any] = {}
    if labels_path.exists():
        labels = json.loads(labels_path.read_text())

    labels[filename] = grid
    labels_path.write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n")
    return {"status": "ok", "saved": filename}
