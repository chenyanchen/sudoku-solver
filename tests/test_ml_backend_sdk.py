from __future__ import annotations

import json

from ml_backend import app as ml_app
from ml_backend.fitters import fit_corners
from ml_backend.schema import extract_filename, extract_image_ref, ordered_corner_labels


KEYPOINT_LABEL_CONFIG = """
<View>
  <KeyPointLabels name="kp-1" toName="img-1">
    <Label value="TL"/>
    <Label value="TR"/>
    <Label value="BR"/>
    <Label value="BL"/>
  </KeyPointLabels>
  <Image name="img-1" value="$img"/>
</View>
"""


def test_extract_image_ref_supports_sdk_parsed_label_config():
    parsed = {
        "kp-1": {
            "type": "KeyPointLabels",
            "to_name": ["img-1"],
            "inputs": [{"type": "Image", "value": "img"}],
            "labels": ["TL", "TR", "BR", "BL"],
            "labels_attrs": {},
        }
    }
    task = {"data": {"img": "/data/upload/6/sample.jpg"}}

    assert extract_image_ref(task, parsed) == "/data/upload/6/sample.jpg"


def test_extract_filename_strips_label_studio_prefix(tmp_path):
    task = {"data": {"img": "/data/upload/6/abc1234-photo.jpg"}}

    assert extract_filename(task, {}, tmp_path) == "photo.jpg"


def test_ordered_corner_labels_normalizes_schema_order():
    parsed = {
        "kp-1": {
            "type": "KeyPointLabels",
            "to_name": ["img-1"],
            "inputs": [{"type": "Image", "value": "img"}],
            "labels": ["BR", "TL", "BL", "TR"],
            "labels_attrs": {},
        }
    }

    assert ordered_corner_labels(parsed) == ["TL", "TR", "BR", "BL"]


def test_predict_route_uses_model_predictions(monkeypatch):
    class FakeModel:
        model_version = "20260401.1"

        def predict(self, tasks, **kwargs):
            assert tasks == [{"data": {"img": "/data/upload/6/sample.jpg"}}]
            return [{"result": [{"x": 1}], "score": 0.4}]

    monkeypatch.setattr(ml_app, "_get_model", lambda payload: FakeModel())

    client = ml_app.app.test_client()
    response = client.post(
        "/predict",
        json={
            "label_config": KEYPOINT_LABEL_CONFIG,
            "tasks": [{"data": {"img": "/data/upload/6/sample.jpg"}}],
        },
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "results": [{"result": [{"x": 1}], "score": 0.4}],
        "model_version": "20260401.1",
    }


def test_get_model_prefers_explicit_label_studio_url_and_refreshes_cached_connection(monkeypatch):
    created: list[object] = []

    class FakeModel:
        def __init__(self, **kwargs):
            self.label_config = kwargs["label_config"]
            self.hostname = kwargs["hostname"]
            self.access_token = kwargs["access_token"]
            self.model_version = "20260401.1"
            created.append(self)

    monkeypatch.setenv("LABEL_STUDIO_URL", "http://localhost:9111")
    monkeypatch.setattr(ml_app, "SudokuMLBackend", FakeModel)
    monkeypatch.setattr(ml_app, "_models", {})

    first = ml_app._get_model(
        {
            "project": {"id": 6},
            "label_config": KEYPOINT_LABEL_CONFIG,
            "hostname": "http://localhost:8080",
            "access_token": "token-a",
        }
    )
    second = ml_app._get_model(
        {
            "project": {"id": 6},
            "label_config": KEYPOINT_LABEL_CONFIG,
            "hostname": "http://host.docker.internal:8080",
            "access_token": "token-b",
        }
    )

    assert len(created) == 1
    assert first is second
    assert second.hostname == "http://localhost:9111"
    assert second.access_token == "token-b"


def test_webhook_route_dispatches_to_model_fit(monkeypatch):
    calls: list[tuple[tuple, dict]] = []

    class FakeModel:
        model_version = "20260401.1"

        def fit(self, *args, **kwargs):
            calls.append((args, kwargs))
            return {"status": "ok", "saved": "IMG_8695.jpg"}

    monkeypatch.setattr(ml_app, "_get_model", lambda payload: FakeModel())

    client = ml_app.app.test_client()
    payload = {"action": "ANNOTATION_CREATED", "annotation": {"id": 1}, "task": {"id": 175}}
    response = client.post("/webhook", json=payload)

    assert response.status_code == 201
    assert response.get_json() == {"status": "ok", "saved": "IMG_8695.jpg"}
    assert calls == [(((),), {"event": "ANNOTATION_CREATED", "data": payload})]


def test_fit_corners_writes_csv(tmp_path):
    repo_root = tmp_path
    image_dir = repo_root / "data" / "raw" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    src_image = image_dir / "IMG_8695.jpg"
    src_image.write_bytes(b"fake-image")

    parsed = {
        "kp-1": {
            "type": "KeyPointLabels",
            "to_name": ["img-1"],
            "inputs": [{"type": "Image", "value": "img"}],
            "labels": ["TL", "TR", "BR", "BL"],
            "labels_attrs": {},
        }
    }
    annotation = {
        "result": [
            {"type": "keypointlabels", "value": {"x": 10, "y": 20, "keypointlabels": ["TL"]}},
            {"type": "keypointlabels", "value": {"x": 80, "y": 20, "keypointlabels": ["TR"]}},
            {"type": "keypointlabels", "value": {"x": 80, "y": 90, "keypointlabels": ["BR"]}},
            {"type": "keypointlabels", "value": {"x": 10, "y": 90, "keypointlabels": ["BL"]}},
        ]
    }
    task = {"data": {"img": "/data/upload/6/abc123-IMG_8695.jpg"}}

    result = fit_corners(annotation, task, parsed, repo_root)

    assert result == {"status": "ok", "saved": "IMG_8695.jpg"}
    csv_path = repo_root / "data" / "corner_cnn" / "labels.csv"
    assert csv_path.exists()
    rows = csv_path.read_text(encoding="utf-8")
    assert "IMG_8695.jpg" in rows
    assert "0.100000" in rows
