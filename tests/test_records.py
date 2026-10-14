"""Public solve-rate records and the upload endpoints that feed them."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api import routes
from backend.main import app
from backend.records import Records

ROOT = Path(__file__).resolve().parents[1]
GRID = [[0] * 9 for _ in range(9)]


def _add(records, image, status):
    return records.add(image, (10, 20), "opencv" if status else None, status,
                       GRID if status else None, 1.0, "v")


def test_stats_count_each_image_by_its_latest_reading(tmp_path):
    records = Records(tmp_path)
    _add(records, b"a", "ambiguous")
    _add(records, b"a", "solved")  # re-uploaded after a model release
    _add(records, b"b", None)      # no grid found
    _add(records, b"c", "solved")
    assert records.stats() == {"solved": 2, "total": 3, "percentage": 66.7}


def test_a_corrected_reading_is_not_solved(tmp_path):
    records = Records(tmp_path)
    token = _add(records, b"a", "solved")
    assert records.correct(token, GRID)  # confirmed unchanged
    assert records.stats()["solved"] == 1
    fixed = [row[:] for row in GRID]
    fixed[0][0] = 5
    assert records.correct(token, fixed)
    assert records.stats()["solved"] == 0
    assert not records.correct("unknown", fixed)


def test_images_are_kept_once_per_content_and_within_budget(tmp_path):
    records = Records(tmp_path, max_image_bytes=10)
    _add(records, b"\xff\xd8\xffyes", "solved")
    _add(records, b"\xff\xd8\xffyes", None)  # the same image again
    assert [p.suffix for p in records.images_dir.iterdir()] == [".jpg"]
    _add(records, b"\x89PNG" + b"x" * 20, "solved")  # over budget
    assert len(list(records.images_dir.iterdir())) == 1


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SUDOKU_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(routes, "_RECORDS", None)
    monkeypatch.setattr(routes, "_RECOGNIZE_LIMIT", routes._RateLimit(3))
    with TestClient(app) as c:
        yield c


def _upload(client, name="IMG_8695.jpg"):
    data = (ROOT / "data/raw/images" / name).read_bytes()
    return client.post("/api/v1/sudoku:recognize", files={"image": (name, data, "image/jpeg")})


def test_upload_is_recorded_and_feedback_counts(client, tmp_path):
    body = _upload(client).json()
    assert body["solved"] and body["token"]
    assert client.get("/api/v1/stats").json() == {"solved": 1, "total": 1, "percentage": 100.0}
    assert len(list((tmp_path / "images").iterdir())) == 1

    labels = json.loads((ROOT / "data/labels/sudoku_labels.json").read_text())
    wrong = [row[:] for row in labels["IMG_8695.jpg"]]
    wrong[0][0] = 0 if wrong[0][0] else 1
    r = client.post("/api/v1/sudoku:feedback", json={"token": body["token"], "grid": {"cells": wrong}})
    assert r.status_code == 200 and r.json() == {"stored": True}
    assert client.get("/api/v1/stats").json()["solved"] == 0
    assert client.post("/api/v1/sudoku:feedback", json={"token": "x", "grid": {"cells": wrong}}).status_code == 404


def test_uploads_are_rate_limited(client):
    for _ in range(3):
        assert _upload(client, "sudoku_7.jpg").status_code == 200
    assert _upload(client, "sudoku_7.jpg").status_code == 429


def test_oversized_upload_is_rejected(client, monkeypatch):
    monkeypatch.setattr(routes, "MAX_UPLOAD_BYTES", 1000)
    assert _upload(client).status_code == 413
