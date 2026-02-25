"""Evaluate CNN OCR against labeled Sudoku images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.api.routes import detect_grid_image, prepare_grid_for_ocr
from backend.cv.cell_extractor import extract_cells
from backend.ocr.cnn_digit_reader import CnnDigitReader
from backend.ocr.digit_reader import DigitReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Sudoku OCR models")
    parser.add_argument(
        "--labels",
        type=Path,
        default=REPO_ROOT / "data" / "sudoku_labels.json",
        help="JSON path: image -> 9x9 grid labels",
    )
    parser.add_argument(
        "--cnn-model",
        type=Path,
        default=REPO_ROOT / "models" / "sudoku_digit_cnn_v1.0.onnx",
        help="ONNX model path",
    )
    parser.add_argument(
        "--with-tesseract-baseline",
        action="store_true",
        help="Evaluate Tesseract baseline in the same report",
    )
    return parser.parse_args()


def load_cells_for_image(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    grid_image = detect_grid_image(image)
    if grid_image is None:
        raise ValueError(f"Failed to detect Sudoku grid: {image_path}")

    ocr_grid = prepare_grid_for_ocr(grid_image)
    return extract_cells(ocr_grid)


def evaluate_single_grid(pred, truth):
    total = 0
    matched = 0
    digit9_total = 0
    digit9_hit = 0
    fp_blank = 0

    for r in range(9):
        for c in range(9):
            t = int(truth[r][c])
            p = int(pred[r][c])
            if t != 0:
                total += 1
                if p == t:
                    matched += 1
                if t == 9:
                    digit9_total += 1
                    if p == 9:
                        digit9_hit += 1
            else:
                if p != 0:
                    fp_blank += 1

    return {
        "givens_total": total,
        "givens_hit": matched,
        "givens_recall": (matched / total) if total else 0.0,
        "digit9_total": digit9_total,
        "digit9_hit": digit9_hit,
        "digit9_recall": (digit9_hit / digit9_total) if digit9_total else 0.0,
        "false_positive_blank": fp_blank,
    }


def merge_metrics(all_metrics):
    agg = {
        "givens_total": 0,
        "givens_hit": 0,
        "digit9_total": 0,
        "digit9_hit": 0,
        "false_positive_blank": 0,
    }

    for m in all_metrics:
        for k in agg:
            agg[k] += m[k]

    agg["givens_recall"] = (
        agg["givens_hit"] / agg["givens_total"] if agg["givens_total"] else 0.0
    )
    agg["digit9_recall"] = (
        agg["digit9_hit"] / agg["digit9_total"] if agg["digit9_total"] else 0.0
    )
    return agg


def main() -> int:
    args = parse_args()

    with args.labels.open("r", encoding="utf-8") as f:
        labels = json.load(f)

    cnn_reader = CnnDigitReader(model_path=args.cnn_model, strict=False)
    if not cnn_reader.is_ready:
        raise RuntimeError(cnn_reader.load_error or "CNN model not ready")

    tess_reader = DigitReader() if args.with_tesseract_baseline else None

    cnn_metrics = []
    tess_metrics = []

    for image_name, truth in labels.items():
        image_path = (REPO_ROOT / image_name).resolve()
        cells = load_cells_for_image(image_path)

        cnn_grid = cnn_reader.recognize_grid(cells)
        cnn_m = evaluate_single_grid(cnn_grid, truth)
        cnn_metrics.append(cnn_m)
        print(f"[CNN] {image_name}: {cnn_m}")

        if tess_reader is not None:
            tess_grid = tess_reader.recognize_grid(cells, threshold=50.0)
            tess_m = evaluate_single_grid(tess_grid, truth)
            tess_metrics.append(tess_m)
            print(f"[TESS] {image_name}: {tess_m}")

    print("\n=== Aggregate ===")
    print("CNN:", merge_metrics(cnn_metrics))
    if tess_metrics:
        print("Tesseract:", merge_metrics(tess_metrics))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
