"""Read every kept upload again with the deployed release, for the public solve rate.

    python -m backend.rescore --version <release>

Needs SUDOKU_DATA_DIR and CNN_MODEL_PATH, as the web app does. Images this
release already read are skipped, so an interrupted run resumes. deploy.sh
runs it after each deploy at the lowest CPU priority, beside the live app.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

from .api.routes import read_image
from .records import Records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True, help="the deployed release, e.g. its git commit")
    args = parser.parse_args()

    records = Records(Path(os.environ["SUDOKU_DATA_DIR"]))
    todo = records.uploaded() - records.rescored(args.version)
    images = sorted(p for p in records.images_dir.iterdir() if p.stem in todo)
    started = time.perf_counter()
    for path in images:
        # Decoded as uploads are (cv2.imdecode applies the EXIF orientation).
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        start = time.perf_counter()
        reading = read_image(image) if image is not None else None
        latency_ms = (time.perf_counter() - start) * 1000.0
        result = reading.recognition if reading else None
        records.add_rescore(
            path.stem, args.version,
            detector=reading.detector_name if reading else None,
            status=result.status if result else None,
            grid=result.grid if result else None,
            latency_ms=latency_ms,
            ocr_model=result.metadata["model_version"] if result else None,
        )
    stats = records.stats()
    print(f"rescored {len(images)} images with {args.version} in {time.perf_counter() - started:.0f} s;"
          f" solved {stats['solved']} of {stats['total']} ({stats['percentage']}%)")


if __name__ == "__main__":
    main()
