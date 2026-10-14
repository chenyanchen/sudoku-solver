"""Evaluate CNN OCR against labeled Sudoku images."""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._paths import DEFAULT_CORNERS, load_corner_labels, resolve_image_path

from backend.api.routes import detect_grid_image
from backend.ocr.pipeline import decode, read_grid
from backend.ocr.cnn_digit_reader import CnnDigitReader, DEFAULT_CNN_MODEL
from backend.cv.detector import warp_grid
from backend.cv.opencv_detector import OpenCVGridDetector
from backend.cv.corner_cnn_detector import CornerCnnDetector
from backend.solver.backtracking import SudokuSolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Sudoku OCR models")
    parser.add_argument(
        "--labels",
        type=Path,
        default=REPO_ROOT / "data" / "labels" / "sudoku_labels.json",
        help="JSON path: image -> 9x9 grid labels",
    )
    parser.add_argument(
        "--cnn-model",
        type=Path,
        default=DEFAULT_CNN_MODEL,
        help="ONNX model path",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON path for aggregate metrics",
    )
    parser.add_argument("--detector", choices=["auto", "gt", "opencv", "corner"], default="auto")
    parser.add_argument("--image-pattern", nargs="+", default=["*"], help="Filename globs, e.g. 'IMG_86*.jpg' 'IMG_870[0-8].jpg'")
    parser.add_argument("--corners", type=Path, default=DEFAULT_CORNERS)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero unless every grid is exact after decode and solved correctly")
    return parser.parse_args()


def load_warped_grid(image_path: Path, detector=None, corners=None):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    if corners is not None:
        h, w = image.shape[:2]
        grid_image = warp_grid(image, corners * np.array([w, h]))
    elif detector is not None:
        detection = detector.detect(image)
        if detection is None:
            raise ValueError(f"Failed to detect Sudoku grid: {image_path}")
        grid_image = warp_grid(image, detection.corners)
    else:
        result = detect_grid_image(image)
        if result is None:
            raise ValueError(f"Failed to detect Sudoku grid: {image_path}")
        grid_image, _ = result

    return grid_image


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
        "grid_exact": pred == truth,
        "givens_total": total,
        "givens_hit": matched,
        "givens_recall": (matched / total) if total else 0.0,
        "digit9_total": digit9_total,
        "digit9_hit": digit9_hit,
        "digit9_recall": (digit9_hit / digit9_total) if digit9_total else 0.0,
        "false_positive_blank": fp_blank,
    }


def merge_metrics(all_metrics):
    agg: dict[str, int | float] = {
        "givens_total": 0,
        "givens_hit": 0,
        "digit9_total": 0,
        "digit9_hit": 0,
        "false_positive_blank": 0,
        "grid_exact": 0,
        "pipeline_exact": 0,
        "solution_exact": 0,
    }

    for m in all_metrics:
        for k in agg:
            agg[k] += m.get(k, 0)

    agg["givens_recall"] = (
        agg["givens_hit"] / agg["givens_total"] if agg["givens_total"] else 0.0
    )
    agg["digit9_recall"] = (
        agg["digit9_hit"] / agg["digit9_total"] if agg["digit9_total"] else 0.0
    )
    agg["images"] = len(all_metrics)
    agg["grid_exact_rate"] = agg["grid_exact"] / len(all_metrics) if all_metrics else 0.0
    return agg


def main() -> int:
    args = parse_args()

    with args.labels.open("r", encoding="utf-8") as f:
        labels = json.load(f)

    cnn_reader = CnnDigitReader(model_path=args.cnn_model)

    labels = {name: grid for name, grid in labels.items() if any(fnmatch.fnmatch(name, pattern) for pattern in args.image_pattern)}
    if not labels:
        raise ValueError("No labeled images match --image-pattern")
    detector = {"opencv": OpenCVGridDetector, "corner": CornerCnnDetector}.get(args.detector)
    detector = detector() if detector else None
    corners_by_name = {}
    if args.detector == "gt":
        corners_by_name = load_corner_labels(args.corners)
    cnn_metrics = []
    per_image = {}
    solver = SudokuSolver()

    skipped = []
    for image_name, truth in labels.items():
        try:
            image_path = resolve_image_path(image_name, args.labels)
        except FileNotFoundError:
            skipped.append(image_name)  # e.g. gitignored data/testset/local/ on another machine
            continue
        try:
            corners = corners_by_name[image_path.name] if args.detector == "gt" else None
            warped = load_warped_grid(image_path, detector, corners)
            # Raw CNN metrics, then the production result after decode().
            cnn_grid, metadata = read_grid(cnn_reader, warped)
            cnn_m = evaluate_single_grid(cnn_grid, truth)
            result = decode(cnn_grid, metadata)
            cnn_m["pipeline_exact"] = result.grid == truth
            # A label with several solutions expects no solution to be shown.
            cnn_m["solution_exact"] = result.solution == solver.solve_unique(truth)[0]
            cnn_m["predicted"] = cnn_grid
        except (ValueError, FileNotFoundError, KeyError) as exc:
            cnn_m = evaluate_single_grid([[0] * 9 for _ in range(9)], truth)
            cnn_m.update(pipeline_exact=False, solution_exact=False, error=str(exc))
        cnn_metrics.append(cnn_m)
        per_image[image_name] = cnn_m
        print(f"[CNN/{args.detector}] {image_name}: " + str({k: v for k, v in cnn_m.items() if k != "predicted"}))

    if skipped:
        print(f"\nSkipped {len(skipped)} labels whose image is missing: {skipped}")
    print("\n=== Aggregate ===")
    aggregate = merge_metrics(cnn_metrics)
    print("CNN:", aggregate)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump({"model": str(args.cnn_model.resolve()), "detector": args.detector, "cnn": aggregate, "per_image": per_image}, f, ensure_ascii=False, indent=2)
        print(f"Saved metrics JSON: {args.output_json}")

    return int(args.strict and (aggregate["pipeline_exact"] != len(cnn_metrics) or aggregate["solution_exact"] != len(cnn_metrics)))


if __name__ == "__main__":
    raise SystemExit(main())
