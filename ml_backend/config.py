from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

_corner_cnn = None
_cnn = None

_CNN_ENV = "CNN_MODEL_PATH"
_CORNER_CNN_ENV = "CORNER_CNN_MODEL"


def validate_backend_model_env() -> tuple[Path, Path]:
    cnn_path = os.getenv(_CNN_ENV)
    corner_cnn_path = os.getenv(_CORNER_CNN_ENV)

    missing = [
        name
        for name, value in ((_CNN_ENV, cnn_path), (_CORNER_CNN_ENV, corner_cnn_path))
        if not value
    ]
    if missing:
        raise RuntimeError(
            "ML Backend requires explicit release model paths: " + ", ".join(missing)
        )

    cnn = Path(cnn_path)
    corner_cnn = Path(corner_cnn_path)
    for path, env_name in ((cnn, _CNN_ENV), (corner_cnn, _CORNER_CNN_ENV)):
        if not path.exists():
            raise RuntimeError(f"{env_name} points to a missing file: {path}")
        meta_path = path.with_suffix(".meta.json")
        if not meta_path.exists():
            raise RuntimeError(
                f"Release metadata file is missing for {env_name}: {meta_path}"
            )

    return cnn, corner_cnn


def read_release_metadata(model_path: Path) -> dict[str, Any]:
    meta_path = model_path.with_suffix(".meta.json")
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Release metadata file is missing: {meta_path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Release metadata file is invalid JSON: {meta_path}"
        ) from exc

    version = (
        payload.get("model_version")
        or payload.get("release_version")
        or payload.get("version")
    )
    if not isinstance(version, str) or not version.strip():
        raise RuntimeError(f"Release metadata file has no model version: {meta_path}")
    payload["model_version"] = version.strip()
    return payload


def current_release_versions() -> tuple[str, str]:
    cnn_path, corner_cnn_path = validate_backend_model_env()
    cnn_version = read_release_metadata(cnn_path)["model_version"]
    corner_cnn_version = read_release_metadata(corner_cnn_path)["model_version"]
    return cnn_version, corner_cnn_version


def get_ls_url() -> str:
    return os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")


def get_ls_access_token() -> str:
    token = os.getenv("LABEL_STUDIO_API_KEY", "")
    if not token:
        return ""
    if not token.startswith("eyJ"):
        return token

    body = json.dumps({"refresh": token}).encode()
    req = urllib.request.Request(
        f"{get_ls_url()}/api/token/refresh", data=body, method="POST"
    )
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["access"]


def get_corner_cnn():
    global _corner_cnn
    if _corner_cnn is None:
        from backend.cv.corner_cnn_detector import CornerCnnDetector

        _corner_cnn = CornerCnnDetector(model_path=os.getenv(_CORNER_CNN_ENV))
    return _corner_cnn


def get_cnn():
    global _cnn
    if _cnn is None:
        cnn_path, _corner_cnn_path = validate_backend_model_env()
        from backend.ocr.cnn_digit_reader import CnnDigitReader

        _cnn = CnnDigitReader(model_path=cnn_path)
    return _cnn
