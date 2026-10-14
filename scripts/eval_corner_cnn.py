"""Evaluate a Corner CNN model against reviewed corners.

Per image: the largest corner error as % of the image's longer side, whether
the prediction passes the grid-regularity gate (as the web chain requires),
and whether the full pipeline solves the puzzle from the raw prediction and
after snapping it to the grid-line lattice.

Usage:
    uv run python scripts/eval_corner_cnn.py --model <candidate.onnx>
    uv run python scripts/eval_corner_cnn.py --labels data/labels/sudoku_labels.json \
        --corners data/labels/sudoku_corners.csv   # training images (optimistic)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.cv.corner_cnn_detector import DEFAULT_CORNER_CNN_MODEL, CornerCnnDetector
from backend.cv.detector import warp_grid
from backend.cv.grid_detector import has_regular_grid_lines, snap_to_lattice
from backend.ocr.cnn_digit_reader import DEFAULT_CNN_MODEL, CnnDigitReader
from backend.ocr.pipeline import recognize
from backend.solver.backtracking import SudokuSolver
from scripts._paths import load_corner_labels, resolve_image_path

TESTSET = REPO_ROOT / "data" / "testset"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Corner CNN model against reviewed corners")
    parser.add_argument("--model", type=Path, default=DEFAULT_CORNER_CNN_MODEL)
    parser.add_argument("--cnn-model", type=Path, default=DEFAULT_CNN_MODEL, help="OCR model")
    parser.add_argument("--labels", type=Path, default=TESTSET / "labels.json")
    parser.add_argument("--corners", type=Path, default=TESTSET / "corners.csv")
    args = parser.parse_args()

    detector = CornerCnnDetector(model_path=str(args.model))
    if not detector.is_ready:
        raise SystemExit(detector.load_error)
    reader = CnnDigitReader(model_path=args.cnn_model)
    truth_corners = load_corner_labels(args.corners)
    solver = SudokuSolver()

    def solves(image: np.ndarray, corners: np.ndarray | None, truth) -> bool:
        if corners is None:
            return False
        solution = recognize(reader, warp_grid(image, corners)).solution
        return solution is not None and solution == solver.solve_unique(truth)[0]

    rows = []
    for name, truth in json.loads(args.labels.read_text()).items():
        try:
            path = resolve_image_path(name, args.labels)
        except FileNotFoundError:
            continue  # gitignored local/ image on another machine
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        predicted = detector.predict(image)
        row = {"image": name, "err": None, "gate": False, "solved": False, "snapped": False}
        if predicted is not None:
            gt = truth_corners[path.name] * [w, h]
            row["err"] = float(np.linalg.norm(predicted - gt, axis=1).max() / max(h, w) * 100)
            row["gate"] = has_regular_grid_lines(cv2.cvtColor(warp_grid(image, predicted), cv2.COLOR_BGR2GRAY))
            row["solved"] = solves(image, predicted, truth)
            snapped = snap_to_lattice(image, predicted, cells=9)
            row["snapped"] = solves(image, None if snapped is None else snapped[1], truth)
        rows.append(row)
        err = "  n/a" if row["err"] is None else f"{row['err']:5.1f}"
        print(f"{name:60s} err%={err} gate={row['gate']!s:5} solved={row['solved']!s:5} snapped={row['snapped']}")

    errs = [r["err"] for r in rows if r["err"] is not None]
    print(
        f"\nimages {len(rows)}  median err {np.median(errs):.2f}%  err<=3% {sum(e <= 3 for e in errs)}  "
        f"gate {sum(r['gate'] for r in rows)}  solved {sum(r['solved'] for r in rows)}  "
        f"solved after snap {sum(r['snapped'] for r in rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
