"""Upload images + model predictions to Label Studio projects.

Supports two project types:
  keypoints - raw images + YOLO corner keypoints (Project 1)
  ocr       - warped grid images + CNN OCR digit grids (Project 2)

Usage:
    export LABEL_STUDIO_API_KEY=<your-token>
    uv run python scripts/upload_to_labelstudio.py --project-type keypoints
    uv run python scripts/upload_to_labelstudio.py --project-type ocr
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

KEYPOINTS_TEMPLATE = """\
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <KeyPointLabels name="corners" toName="image" smart="true">
    <Label value="TL" background="red"/>
    <Label value="TR" background="green"/>
    <Label value="BR" background="blue"/>
    <Label value="BL" background="yellow"/>
  </KeyPointLabels>
</View>"""

OCR_TEMPLATE = """\
<View>
  <View style="display:flex;flex-direction:row;gap:16px">
    <View style="flex:1">
      <Image name="image" value="$image" zoom="true" zoomControl="true"/>
    </View>
    <View style="flex:1;font-family:monospace;font-size:16px;white-space:pre-line;\
padding:12px;background:#f5f5f5;border-radius:8px">
      <Header value="OCR Result (verify against image)"/>
      <Text name="ocr_ref" value="$ocr_grid"/>
    </View>
  </View>
  <Header value="Corrected Grid (9 rows, 0=empty, space-separated)"/>
  <TextArea name="grid" toName="image" rows="9" maxSubmissions="1" editable="true"\
 placeholder="Only fill if OCR result above is wrong"/>
</View>"""


def _resolve_api_token(url: str, api_key: str) -> tuple[str, str]:
    """Resolve auth header. If api_key is a JWT refresh token, exchange for access token."""
    if not api_key.startswith("eyJ"):
        return "Token", api_key
    # Try as access token first.
    try:
        req = urllib.request.Request(f"{url}/api/projects")
        req.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(req):
            return "Bearer", api_key
    except urllib.error.HTTPError:
        pass
    # Exchange refresh token for access token.
    body = json.dumps({"refresh": api_key}).encode()
    req = urllib.request.Request(f"{url}/api/token/refresh", data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return "Bearer", data["access"]


class LabelStudioClient:
    """Minimal Label Studio API client using stdlib only."""

    def __init__(self, url: str, api_key: str) -> None:
        self.url = url.rstrip("/")
        self._auth_prefix, self._token = _resolve_api_token(self.url, api_key)

    def _request(self, method: str, path: str, data: dict | None = None) -> dict | list:
        url = f"{self.url}{path}"
        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(url, data=body, method=method)
        req.add_header("Authorization", f"{self._auth_prefix} {self._token}")
        if body:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            err_body = e.read().decode()
            raise RuntimeError(f"Label Studio API {method} {path}: {e.code} {err_body}") from e

    def get(self, path: str) -> dict | list:
        return self._request("GET", path)

    def post(self, path: str, data: dict) -> dict:
        return self._request("POST", path, data)

    def patch(self, path: str, data: dict) -> dict:
        return self._request("PATCH", path, data)

    def find_or_create_project(self, name: str, label_config: str) -> int:
        projects = self.get("/api/projects")
        results = projects.get("results", projects) if isinstance(projects, dict) else projects
        for p in results:
            if p.get("title") == name:
                return p["id"]
        result = self.post("/api/projects", {"title": name, "label_config": label_config})
        print(f"  Created project: {name} (id={result['id']})")
        return result["id"]

    def get_existing_tasks(self, project_id: int) -> dict[str, int]:
        """Return {external_id: task_id} for all tasks in a project."""
        mapping: dict[str, int] = {}
        page = 1
        while True:
            resp = self.get(f"/api/tasks?project={project_id}&page={page}&page_size=100")
            tasks = resp.get("tasks", resp) if isinstance(resp, dict) else resp
            for t in tasks:
                ext_id = (t.get("meta") or {}).get("external_id", "")
                if ext_id:
                    mapping[ext_id] = t["id"]
            if isinstance(resp, dict) and resp.get("next"):
                page += 1
            else:
                break
        return mapping

    def create_task(self, project_id: int, data: dict, external_id: str) -> int:
        result = self.post(
            "/api/tasks",
            {"data": data, "meta": {"external_id": external_id}, "project": project_id},
        )
        return result["id"]

    def create_prediction(
        self, task_id: int, result: list, model_version: str, score: float = 0.0,
    ) -> None:
        # Set score on each result item so Label Studio computes avg_predictions_score.
        for item in result:
            item.setdefault("score", score)
        self.post("/api/predictions", {
            "task": task_id,
            "result": result,
            "model_version": model_version,
            "score": score,
        })


def _image_to_data_uri(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode()
    suffix = path.suffix.lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        suffix.lstrip("."), "image/jpeg"
    )
    return f"data:{mime};base64,{b64}"


def upload_keypoints(client: LabelStudioClient, src: Path) -> None:
    """Upload raw images + YOLO corner predictions to keypoints project."""
    project_id = client.find_or_create_project("Sudoku Grid Keypoints", KEYPOINTS_TEMPLATE)
    existing = client.get_existing_tasks(project_id)

    warped_dir = REPO_ROOT / "data" / "warped_grids"

    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(src.glob(ext))

    created = 0
    skipped = 0
    for img_path in sorted(images):
        ext_id = img_path.name
        if ext_id in existing:
            skipped += 1
            continue

        data_uri = _image_to_data_uri(img_path)

        # Read OCR data for priority score (reflects solvability).
        ocr_file = warped_dir / f"{img_path.stem}_ocr.json"
        score = 0.0
        if ocr_file.exists():
            ocr_info = json.loads(ocr_file.read_text())
            score = ocr_info.get("priority_score", 0.0)

        task_data = {"image": data_uri, "score": score}
        task_id = client.create_task(project_id, task_data, ext_id)

        # Upload YOLO corner prediction if available.
        corners_file = warped_dir / f"{img_path.stem}_corners.json"
        if corners_file.exists():
            corners_data = json.loads(corners_file.read_text())
            corners = corners_data["corners"]
            img_w = corners_data["image_width"]
            img_h = corners_data["image_height"]
            conf = corners_data["confidence"]

            labels = ["TL", "TR", "BR", "BL"]
            result = []
            for i, label in enumerate(labels):
                x_pct = corners[i][0] / img_w * 100
                y_pct = corners[i][1] / img_h * 100
                result.append({
                    "from_name": "corners",
                    "to_name": "image",
                    "type": "keypointlabels",
                    "value": {
                        "x": round(x_pct, 4),
                        "y": round(y_pct, 4),
                        "width": 0.3,
                        "keypointlabels": [label],
                    },
                })
            client.create_prediction(task_id, result, "yolo_grid_detect", conf)

        print(f"  ADD {ext_id} (task={task_id})")
        created += 1

    print(f"\nKeypoints: created={created}, skipped={skipped}")


def upload_ocr(client: LabelStudioClient, src: Path) -> None:
    """Upload warped grid images + OCR predictions to OCR project."""
    project_id = client.find_or_create_project("Sudoku Digit OCR", OCR_TEMPLATE)
    existing = client.get_existing_tasks(project_id)

    manifest_path = src / "manifest.json"
    if not manifest_path.exists():
        print("ERROR: Run generate_warped_grids.py first", file=sys.stderr)
        return
    manifest = json.loads(manifest_path.read_text())

    created = 0
    skipped = 0
    for warped_name, original_name in sorted(manifest.items()):
        ext_id = original_name  # use original filename as external ID
        if ext_id in existing:
            skipped += 1
            continue

        warped_path = src / warped_name
        if not warped_path.exists():
            continue

        stem = warped_path.stem.removesuffix("_warped")
        ocr_file = src / f"{stem}_ocr.json"
        ocr_grid_text = ""
        ocr_grid = [[0] * 9 for _ in range(9)]
        priority_score = 0.0
        solved = False
        if ocr_file.exists():
            ocr_data = json.loads(ocr_file.read_text())
            ocr_grid = ocr_data["grid"]
            solved = ocr_data.get("solved", False)
            priority_score = ocr_data.get("priority_score", 0.0)
            ocr_grid_text = "\n".join(
                " ".join(str(d) for d in row) for row in ocr_grid
            )

        status_tag = "SOLVED" if solved else "NEEDS REVIEW"
        data_uri = _image_to_data_uri(warped_path)
        task_data = {
            "image": data_uri,
            "ocr_grid": f"[{status_tag}] score={priority_score}\n\n{ocr_grid_text}",
            "score": priority_score,
        }
        task_id = client.create_task(project_id, task_data, ext_id)

        # Upload OCR prediction with priority score for sorting.
        result = [{
            "from_name": "grid",
            "to_name": "image",
            "type": "textarea",
            "value": {"text": [
                " ".join(str(d) for d in row) for row in ocr_grid
            ]},
        }]
        client.create_prediction(task_id, result, "cnn_ocr", score=priority_score)

        print(f"  ADD {original_name} (task={task_id})")
        created += 1

    print(f"\nOCR: created={created}, skipped={skipped}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload to Label Studio")
    parser.add_argument(
        "--project-type", required=True, choices=["keypoints", "ocr"],
    )
    parser.add_argument("--src", type=Path, default=None)
    parser.add_argument("--url", type=str, default=os.getenv("LABEL_STUDIO_URL", "http://localhost:8080"))
    parser.add_argument("--api-key", type=str, default=os.getenv("LABEL_STUDIO_API_KEY"))
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: Set LABEL_STUDIO_API_KEY env var or pass --api-key", file=sys.stderr)
        return 1

    client = LabelStudioClient(args.url, args.api_key)

    if args.project_type == "keypoints":
        src = args.src or (REPO_ROOT / "data" / "raw" / "images")
        upload_keypoints(client, src)
    else:
        src = args.src or (REPO_ROOT / "data" / "warped_grids")
        upload_ocr(client, src)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
