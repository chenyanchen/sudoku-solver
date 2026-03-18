"""Export reviewed annotations from Label Studio back to repo formats.

Supports two project types:
  keypoints - Export corner annotations to YOLO pose label files
  ocr       - Export digit grid corrections into sudoku_labels.json

Usage:
    export LABEL_STUDIO_API_KEY=<your-token>
    uv run python scripts/export_from_labelstudio.py --project-type keypoints
    uv run python scripts/export_from_labelstudio.py --project-type ocr --merge
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.auto_label_yolo import corners_to_yolo_pose


class LabelStudioClient:
    """Minimal Label Studio API client using stdlib only."""

    def __init__(self, url: str, api_key: str) -> None:
        self.url = url.rstrip("/")
        # Reuse token resolution from upload script.
        from scripts.upload_to_labelstudio import _resolve_api_token
        self._auth_prefix, self._token = _resolve_api_token(self.url, api_key)

    def get(self, path: str) -> dict | list:
        url = f"{self.url}{path}"
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"{self._auth_prefix} {self._token}")
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            err_body = e.read().decode()
            raise RuntimeError(f"Label Studio API GET {path}: {e.code} {err_body}") from e

    def find_project_by_name(self, name: str) -> int | None:
        projects = self.get("/api/projects")
        results = projects.get("results", projects) if isinstance(projects, dict) else projects
        for p in results:
            if p.get("title") == name:
                return p["id"]
        return None

    def export_annotations(self, project_id: int) -> list[dict]:
        return self.get(f"/api/projects/{project_id}/export?exportType=JSON")


def _extract_external_id(task: dict) -> str:
    """Get original filename from task metadata."""
    return (task.get("meta") or {}).get("external_id", "")


def export_keypoints(client: LabelStudioClient, project_id: int, output_dir: Path) -> None:
    """Export keypoint annotations to YOLO pose label format."""
    tasks = client.export_annotations(project_id)

    img_dir = output_dir / "images" / "train"
    lbl_dir = output_dir / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    raw_images_dir = REPO_ROOT / "data" / "raw" / "images"
    exported = 0
    skipped = 0

    for task in tasks:
        annotations = task.get("annotations", [])
        if not annotations:
            skipped += 1
            continue

        # Use the latest annotation.
        annotation = annotations[-1]
        if annotation.get("was_cancelled"):
            skipped += 1
            continue

        ext_id = _extract_external_id(task)
        if not ext_id:
            skipped += 1
            continue

        # Parse keypoint results.
        results = annotation.get("result", [])
        corner_map: dict[str, tuple[float, float]] = {}
        for r in results:
            if r.get("type") != "keypointlabels":
                continue
            value = r["value"]
            labels = value.get("keypointlabels", [])
            if not labels:
                continue
            label = labels[0]
            # Label Studio keypoints use percentage coordinates (0-100).
            x_pct = value["x"]
            y_pct = value["y"]
            corner_map[label] = (x_pct, y_pct)

        if len(corner_map) < 4:
            print(f"  SKIP {ext_id}: only {len(corner_map)} corners annotated")
            skipped += 1
            continue

        # Load original image to get dimensions.
        src_image = raw_images_dir / ext_id
        if not src_image.exists():
            print(f"  SKIP {ext_id}: source image not found")
            skipped += 1
            continue

        image = cv2.imread(str(src_image))
        if image is None:
            skipped += 1
            continue
        h, w = image.shape[:2]

        # Convert percentage to absolute coords, order: TL, TR, BR, BL.
        corners = []
        for label in ["TL", "TR", "BR", "BL"]:
            x_pct, y_pct = corner_map[label]
            corners.append([x_pct / 100.0 * w, y_pct / 100.0 * h])
        corners_np = np.float32(corners)

        # Write YOLO pose label.
        label_line = corners_to_yolo_pose(corners_np, w, h)
        stem = Path(ext_id).stem
        (lbl_dir / f"{stem}.txt").write_text(label_line + "\n")

        # Copy image if not already there.
        dst_img = img_dir / ext_id
        if not dst_img.exists():
            shutil.copy2(src_image, dst_img)

        print(f"  OK  {ext_id}")
        exported += 1

    print(f"\nKeypoints export: {exported} exported, {skipped} skipped")


def export_ocr(client: LabelStudioClient, project_id: int, merge: bool) -> None:
    """Export OCR annotations to sudoku_labels.json."""
    tasks = client.export_annotations(project_id)

    labels_path = REPO_ROOT / "data" / "labels" / "sudoku_labels.json"
    sources_path = REPO_ROOT / "data" / "labels" / "sudoku_label_sources.json"

    existing_labels: dict = {}
    if merge and labels_path.exists():
        existing_labels = json.loads(labels_path.read_text())

    existing_sources: dict = {}
    if sources_path.exists():
        existing_sources = json.loads(sources_path.read_text())

    updated = 0
    skipped = 0

    for task in tasks:
        annotations = task.get("annotations", [])
        if not annotations:
            skipped += 1
            continue

        annotation = annotations[-1]
        if annotation.get("was_cancelled"):
            skipped += 1
            continue

        ext_id = _extract_external_id(task)
        if not ext_id:
            skipped += 1
            continue

        # Parse TextArea results.
        results = annotation.get("result", [])
        grid_text_lines: list[str] = []
        for r in results:
            if r.get("from_name") == "grid" and r.get("type") == "textarea":
                grid_text_lines = r["value"].get("text", [])
                break

        if not grid_text_lines:
            skipped += 1
            continue

        # Parse grid: each line is 9 space-separated digits.
        grid: list[list[int]] = []
        for line in grid_text_lines:
            row = [int(d) for d in line.strip().split()]
            if len(row) != 9:
                print(f"  WARN {ext_id}: row has {len(row)} digits, expected 9")
                break
            grid.append(row)

        if len(grid) != 9:
            print(f"  SKIP {ext_id}: grid has {len(grid)} rows, expected 9")
            skipped += 1
            continue

        existing_labels[ext_id] = grid
        existing_sources[ext_id] = {
            "source": "label_studio",
            "annotator": str(annotation.get("completed_by", "")),
        }
        print(f"  OK  {ext_id}")
        updated += 1

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(existing_labels, indent=2, ensure_ascii=False) + "\n")
    sources_path.write_text(json.dumps(existing_sources, indent=2, ensure_ascii=False) + "\n")

    print(f"\nOCR export: {updated} updated, {skipped} skipped")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export from Label Studio")
    parser.add_argument("--project-type", required=True, choices=["keypoints", "ocr"])
    parser.add_argument("--project-id", type=int, default=None, help="Label Studio project ID (auto-detected by name if omitted)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--merge", action="store_true", help="Merge into existing labels (OCR only)")
    parser.add_argument("--url", type=str, default=os.getenv("LABEL_STUDIO_URL", "http://localhost:8080"))
    parser.add_argument("--api-key", type=str, default=os.getenv("LABEL_STUDIO_API_KEY"))
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: Set LABEL_STUDIO_API_KEY env var or pass --api-key", file=sys.stderr)
        return 1

    client = LabelStudioClient(args.url, args.api_key)

    if args.project_type == "keypoints":
        project_name = "Sudoku Grid Keypoints"
        pid = args.project_id or client.find_project_by_name(project_name)
        if pid is None:
            print(f"ERROR: Project '{project_name}' not found", file=sys.stderr)
            return 1
        output = args.output or (REPO_ROOT / "data" / "yolo_grid")
        export_keypoints(client, pid, output)
    else:
        project_name = "Sudoku Digit OCR"
        pid = args.project_id or client.find_project_by_name(project_name)
        if pid is None:
            print(f"ERROR: Project '{project_name}' not found", file=sys.stderr)
            return 1
        export_ocr(client, pid, args.merge)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
