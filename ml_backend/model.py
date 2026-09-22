from __future__ import annotations

import logging
import urllib.request
from typing import Any

from label_studio_ml.model import LabelStudioMLBase

from .config import (
    REPO_ROOT,
    current_release_versions,
    get_cnn,
    get_corner_cnn,
    get_ls_access_token,
    get_ls_url,
    validate_backend_model_env,
)
from .fitters import fit_corners, fit_ocr
from .predictors import load_task_image, predict_corners, predict_ocr
from .schema import is_keypoints_project

logger = logging.getLogger("ml_backend")


class SudokuMLBackend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        validate_backend_model_env()
        self.repo_root = REPO_ROOT
        self.corner_cnn = get_corner_cnn()
        self.cnn = get_cnn()
        self.model_version = self._compute_model_version()

    def _compute_model_version(self) -> str:
        cnn_version, corner_version = current_release_versions()
        if is_keypoints_project(self.parsed_label_config):
            return corner_version
        return f"{cnn_version}+{corner_version}"

    def predict(self, tasks, **kwargs):
        ls_url = self.hostname or get_ls_url()
        ls_token = self.access_token or get_ls_access_token()
        predictions = []

        for task in tasks or []:
            image_ref, image = load_task_image(task, self.parsed_label_config, ls_url, ls_token)
            if image is None:
                logger.warning("predict: empty or unreadable image ref: %r", image_ref)
                predictions.append({"result": [], "score": 0.0})
                continue

            if is_keypoints_project(self.parsed_label_config):
                result, score = predict_corners(image, self.parsed_label_config, self.corner_cnn)
            else:
                result, score = predict_ocr(
                    image, self.parsed_label_config, self.corner_cnn, self.cnn
                )
            predictions.append({"result": result, "score": score})

        return predictions

    def _fetch_json(self, path: str) -> dict[str, Any]:
        token = self.access_token or get_ls_access_token()
        req = urllib.request.Request(f"{self.hostname or get_ls_url()}{path}")
        if token:
            prefix = "Bearer" if token.startswith("eyJ") else "Token"
            req.add_header("Authorization", f"{prefix} {token}")
        with urllib.request.urlopen(req) as resp:
            import json

            return json.loads(resp.read())

    def _resolve_webhook_payload(self, data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        annotation = data.get("annotation") or {}
        task = data.get("task") or annotation.get("task") or {}

        if isinstance(task, int):
            try:
                task = self._fetch_json(f"/api/tasks/{task}")
            except Exception:
                task = {}

        if not annotation and data.get("annotation_id"):
            try:
                annotation = self._fetch_json(f"/api/annotations/{data['annotation_id']}")
            except Exception:
                annotation = {}

        return annotation, task

    def fit(self, tasks, workdir=None, **kwargs):
        event = kwargs.get("event")
        data = kwargs.get("data") or {}

        if event:
            annotation, task = self._resolve_webhook_payload(data)
        else:
            annotation = data.get("annotation", {}) if data else {}
            task = data.get("task", annotation.get("task", {})) if data else {}

        if not annotation or not task or isinstance(task, int):
            logger.warning("fit: insufficient annotation/task payload for event=%s", event)
            return {"status": "skipped"}

        if is_keypoints_project(self.parsed_label_config):
            result = fit_corners(annotation, task, self.parsed_label_config, self.repo_root)
        else:
            result = fit_ocr(annotation, task, self.parsed_label_config, self.repo_root)

        logger.info("fit: event=%s result=%s", event or "manual", result)
        return result
