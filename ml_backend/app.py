"""SDK-based Label Studio ML backend server."""

from __future__ import annotations

import logging
from typing import Any

from flask import Flask, jsonify, request

from .config import get_ls_access_token, get_ls_url
from .model import SudokuMLBackend
from .schema import extract_filename

logger = logging.getLogger("ml_backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

app = Flask(__name__)
_models: dict[str, SudokuMLBackend] = {}


def _project_key(project: Any) -> str:
    if isinstance(project, dict):
        return str(project.get("id") or project.get("pk") or "default")
    if project is None:
        return "default"
    return str(project)


def _get_label_config(payload: dict[str, Any]) -> str:
    return (
        payload.get("label_config")
        or payload.get("schema")
        or payload.get("project", {}).get("label_config", "")
        or ""
    )


def _resolve_hostname(payload: dict[str, Any]) -> str:
    configured = get_ls_url()
    if configured:
        return configured
    return payload.get("hostname") or ""


def _get_model(payload: dict[str, Any]) -> SudokuMLBackend:
    project_key = _project_key(payload.get("project"))
    label_config = _get_label_config(payload)
    hostname = _resolve_hostname(payload)
    access_token = payload.get("access_token") or get_ls_access_token()

    model = _models.get(project_key)
    if model is None or (label_config and model.label_config != label_config):
        model = SudokuMLBackend(
            label_config=label_config,
            hostname=hostname,
            access_token=access_token,
        )
        _models[project_key] = model
    else:
        model.hostname = hostname
        model.access_token = access_token
    return model


@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify({"status": "UP", "model_class": "SudokuMLBackend"})


@app.route("/setup", methods=["POST"])
def setup() -> Any:
    payload = request.get_json(silent=True) or {}
    model = _get_model(payload)
    return jsonify({"model_version": model.model_version})


@app.route("/predict", methods=["POST"])
def predict() -> Any:
    payload = request.get_json(silent=True) or {}
    model = _get_model(payload)
    params = payload.get("params") or {}
    predictions = model.predict(payload.get("tasks", []), **params)
    return jsonify({"results": predictions, "model_version": model.model_version})


@app.route("/webhook", methods=["POST"])
def webhook() -> Any:
    payload = request.get_json(silent=True) or {}
    model = _get_model(payload)
    event = payload.get("action")
    result = model.fit((), event=event, data=payload)
    return jsonify(result), 201


@app.route("/fit", methods=["POST"])
def fit() -> Any:
    payload = request.get_json(silent=True) or {}
    model = _get_model(payload)
    result = model.fit((), data=payload)
    return jsonify(result)


__all__ = ["app", "extract_filename"]
