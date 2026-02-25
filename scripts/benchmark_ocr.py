"""Benchmark OCR runtime before/after rescue variants."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.api.routes import detect_grid_image, prepare_grid_for_ocr
from backend.cv.cell_extractor import clean_cell, extract_cells, extract_digit
from backend.ocr.digit_reader import DigitReader


class BaselineDigitReader(DigitReader):
    """Legacy OCR pipeline without rescue variants."""

    def recognize_digit(self, cell_image, threshold: float = 30.0):
        if cell_image is None or cell_image.size == 0:
            return None

        if self._is_likely_blank(cell_image):
            return None

        normalized = clean_cell(cell_image, cell_size=48)
        digit_image, has_digit = extract_digit(normalized)
        candidate = digit_image if has_digit else normalized

        for processed in self._prepare_fast_variants(candidate):
            result = self._run_ocr(processed, threshold)
            if result is not None:
                return result

        if not has_digit:
            return None

        for processed in self._prepare_fallback_variants(digit_image):
            result = self._run_ocr(processed, threshold)
            if result is not None:
                return result

        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Sudoku OCR runtime")
    parser.add_argument(
        "--images",
        nargs="+",
        default=["sudoku_2.png", "sudoku_3.png"],
        help="Image paths to benchmark",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of rounds for each implementation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="OCR confidence threshold",
    )
    return parser.parse_args()


def resolve_image_path(image_arg: str) -> Path:
    candidate = Path(image_arg)
    if candidate.is_file():
        return candidate

    from_repo = REPO_ROOT / image_arg
    if from_repo.is_file():
        return from_repo

    raise FileNotFoundError(f"Image not found: {image_arg}")


def load_cells(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to decode image: {image_path}")

    grid_image = detect_grid_image(image)
    if grid_image is None:
        raise ValueError(f"Failed to detect Sudoku grid: {image_path}")

    ocr_grid = prepare_grid_for_ocr(grid_image)
    return extract_cells(ocr_grid)


def run_benchmark(reader: DigitReader, cells_by_image, rounds: int, threshold: float):
    start = time.perf_counter()

    for _ in range(rounds):
        for cells in cells_by_image.values():
            reader.recognize_grid(cells, threshold=threshold)

    elapsed = time.perf_counter() - start
    image_count = len(cells_by_image)
    avg_per_image = elapsed / (rounds * image_count)
    return elapsed, avg_per_image


def main() -> int:
    args = parse_args()

    image_paths = [resolve_image_path(image) for image in args.images]
    cells_by_image = {str(path): load_cells(path) for path in image_paths}

    baseline_reader = BaselineDigitReader()
    optimized_reader = DigitReader()

    baseline_total, baseline_avg = run_benchmark(
        baseline_reader, cells_by_image, args.rounds, args.threshold
    )
    optimized_total, optimized_avg = run_benchmark(
        optimized_reader, cells_by_image, args.rounds, args.threshold
    )

    overhead_ratio = ((optimized_avg - baseline_avg) / baseline_avg) * 100.0

    print("OCR benchmark results")
    print(f"images={len(image_paths)} rounds={args.rounds} threshold={args.threshold}")
    print(f"baseline_total={baseline_total:.3f}s baseline_avg_per_image={baseline_avg:.3f}s")
    print(
        f"optimized_total={optimized_total:.3f}s "
        f"optimized_avg_per_image={optimized_avg:.3f}s"
    )
    print(f"overhead={overhead_ratio:.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
