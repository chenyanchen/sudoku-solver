"""Record of public recognitions: the site's solve rate and new training data.

Each upload is a reading. Stats count distinct images (by content hash) by
their latest reading, so re-uploading an image after a model release shows
the improvement. A reading counts as solved when it produced a unique
solution and the uploader did not correct its givens.

Every uploaded image is kept (once per content hash, within a disk budget)
as training data; readings hold no personal data (no IP, no file name).
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY,
    token TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    image_sha256 TEXT NOT NULL,
    image_bytes INTEGER NOT NULL,
    width INTEGER,
    height INTEGER,
    detector TEXT,             -- NULL: no grid found
    status TEXT,               -- solved / ambiguous / unsolvable; NULL: no grid found
    grid TEXT,                 -- recognized givens, JSON
    latency_ms REAL NOT NULL,
    ocr_model TEXT,
    image_kept INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS readings_image ON readings (image_sha256);
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY,
    reading_id INTEGER NOT NULL REFERENCES readings (id),
    created_at TEXT NOT NULL,
    grid TEXT NOT NULL         -- givens as corrected by the uploader, JSON
);
"""

_IMAGE_TYPES = {b"\xff\xd8\xff": ".jpg", b"\x89PNG": ".png", b"GIF8": ".gif", b"BM": ".bmp"}


def _extension(data: bytes) -> str:
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return next((ext for magic, ext in _IMAGE_TYPES.items() if data.startswith(magic)), ".img")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Records:
    def __init__(self, data_dir: Path, max_image_bytes: int = 5 * 1024**3):
        self.images_dir = data_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.max_image_bytes = max_image_bytes
        self._image_bytes = sum(p.stat().st_size for p in self.images_dir.iterdir())
        self._db = sqlite3.connect(data_dir / "records.sqlite3", check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._lock = threading.Lock()

    def add(
        self,
        image: bytes,
        shape: tuple[int, int] | None,
        detector: str | None,
        status: str | None,
        grid: list[list[int]] | None,
        latency_ms: float,
        ocr_model: str | None,
    ) -> str:
        """Store a reading; returns the token that authorizes its correction."""
        sha = hashlib.sha256(image).hexdigest()
        token = secrets.token_urlsafe(16)
        with self._lock:
            kept = self._keep(sha, image)
            self._db.execute(
                "INSERT INTO readings (token, created_at, image_sha256, image_bytes, width, height, detector,"
                " status, grid, latency_ms, ocr_model, image_kept) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (token, _now(), sha, len(image), shape[1] if shape else None, shape[0] if shape else None,
                 detector, status, json.dumps(grid) if grid else None, latency_ms, ocr_model, int(kept)),
            )
            self._db.commit()
        return token

    def _keep(self, sha: str, image: bytes) -> bool:
        path = self.images_dir / f"{sha}{_extension(image)}"
        if path.exists():
            return True
        if self._image_bytes + len(image) > self.max_image_bytes:
            return False
        path.write_bytes(image)
        self._image_bytes += len(image)
        return True

    def correct(self, token: str, grid: list[list[int]]) -> bool:
        """Store the uploader's corrected givens; False for an unknown token."""
        with self._lock:
            row = self._db.execute("SELECT id FROM readings WHERE token = ?", (token,)).fetchone()
            if row is None:
                return False
            self._db.execute(
                "INSERT INTO corrections (reading_id, created_at, grid) VALUES (?, ?, ?)",
                (row[0], _now(), json.dumps(grid)),
            )
            self._db.commit()
        return True

    def stats(self) -> dict[str, float | int]:
        with self._lock:
            total, solved = self._db.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(
                    r.status = 'solved' AND NOT EXISTS (
                        SELECT 1 FROM corrections c WHERE c.reading_id = r.id AND c.grid != r.grid
                    )
                ), 0)
                FROM readings r
                WHERE r.id IN (SELECT MAX(id) FROM readings GROUP BY image_sha256)
                """
            ).fetchone()
        return {"solved": solved, "total": total, "percentage": round(100 * solved / total, 1) if total else 0.0}
