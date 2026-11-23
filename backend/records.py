"""Record of public recognitions: the site's solve rate and new training data.

Each upload is a reading. Stats count distinct images (by content hash) by
their latest outcome: the latest reading, or a rescore if newer. After each
deploy every kept image is read again (`backend.rescore`), so the rate is
what the deployed version solves, not what earlier versions did. A reading
counts as solved when its givens have a solution (the site shows one) and
the uploader did not correct them; a rescore when its givens have one.

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
    status TEXT,               -- unique / ambiguous / unsolvable; NULL: no grid found
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
CREATE TABLE IF NOT EXISTS rescores (
    id INTEGER PRIMARY KEY,
    image_sha256 TEXT NOT NULL,
    reading_id INTEGER NOT NULL REFERENCES readings (id),  -- the image's latest reading then
    created_at TEXT NOT NULL,
    version TEXT NOT NULL,     -- the deployed release that read it
    detector TEXT,             -- NULL: no grid found
    status TEXT,               -- unique / ambiguous / unsolvable; NULL: no grid found
    grid TEXT,                 -- recognized givens, JSON
    latency_ms REAL NOT NULL,
    ocr_model TEXT
);
CREATE INDEX IF NOT EXISTS rescores_image ON rescores (image_sha256);
"""

# No puzzle with fewer givens has a unique solution, so none is published.
MIN_PUZZLE_GIVENS = 17

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
        # Readings before 2026-10-15 called a unique reading 'solved'.
        self._db.execute("UPDATE readings SET status = 'unique' WHERE status = 'solved'")
        self._db.commit()
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

    def add_rescore(
        self,
        sha: str,
        version: str,
        detector: str | None,
        status: str | None,
        grid: list[list[int]] | None,
        latency_ms: float,
        ocr_model: str | None,
    ) -> None:
        """Store a kept image's reading by the deployed release."""
        with self._lock:
            self._db.execute(
                "INSERT INTO rescores (image_sha256, reading_id, created_at, version, detector, status,"
                " grid, latency_ms, ocr_model) VALUES"
                " (?, (SELECT MAX(id) FROM readings WHERE image_sha256 = ?), ?, ?, ?, ?, ?, ?, ?)",
                (sha, sha, _now(), version, detector, status, json.dumps(grid) if grid else None,
                 latency_ms, ocr_model),
            )
            self._db.commit()

    def uploaded(self) -> set[str]:
        """Every image with a reading."""
        with self._lock:
            return {row[0] for row in self._db.execute("SELECT DISTINCT image_sha256 FROM readings")}

    def rescored(self, version: str) -> set[str]:
        """Images already read by `version`."""
        with self._lock:
            rows = self._db.execute("SELECT DISTINCT image_sha256 FROM rescores WHERE version = ?", (version,))
            return {row[0] for row in rows}

    def stats(self) -> dict[str, float | int]:
        """Images whose latest outcome solved: the latest reading, uncorrected,
        or a newer rescore. An ambiguous reading counts when it has a puzzle's
        worth of givens: some published puzzles are ambiguous, but a partial
        grid, or no Sudoku at all, has solutions too. Corrections attach to an
        upload's reading only; the deployed release rereads the image itself.
        """
        def solves(t: str) -> str:
            return f"""({t}.status = 'unique' OR ({t}.status = 'ambiguous' AND (
                SELECT COUNT(*) FROM json_each({t}.grid) AS row, json_each(row.value) AS cell
                WHERE cell.value != 0
            ) >= {MIN_PUZZLE_GIVENS}))"""

        with self._lock:
            total, solved = self._db.execute(
                f"""
                SELECT COUNT(*), COALESCE(SUM(CASE
                    WHEN s.reading_id >= r.id THEN {solves("s")}
                    ELSE {solves("r")} AND NOT EXISTS (
                        SELECT 1 FROM corrections c WHERE c.reading_id = r.id AND c.grid != r.grid
                    )
                END), 0)
                FROM readings r
                LEFT JOIN rescores s
                    ON s.id = (SELECT MAX(id) FROM rescores WHERE image_sha256 = r.image_sha256)
                WHERE r.id IN (SELECT MAX(id) FROM readings GROUP BY image_sha256)
                """,
            ).fetchone()
        return {"solved": solved, "total": total, "percentage": round(100 * solved / total, 1) if total else 0.0}
