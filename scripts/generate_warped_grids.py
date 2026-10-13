"""Generate warped 450x450 grids + OCR pre-annotations from raw images.

Produces intermediate data consumed by the Label Studio upload script.

Usage:
    uv run python scripts/generate_warped_grids.py
    uv run python scripts/generate_warped_grids.py --yolo-model models/releases/yolo_grid_detect.onnx
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.cv.yolo_detector import YoloGridDetector
from backend.ocr.cnn_digit_reader import CnnDigitReader
from backend.ocr.pipeline import recognize


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate warped grids + OCR pre-annotations")
    parser.add_argument("--src", type=Path, default=REPO_ROOT / "data" / "raw" / "images")
    parser.add_argument("--dst", type=Path, default=REPO_ROOT / "data" / "warped_grids")
    parser.add_argument("--yolo-model", type=str, default=None)
    parser.add_argument("--cnn-model", type=str, default=os.getenv("CNN_MODEL_PATH"))
    args = parser.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)

    detector = YoloGridDetector(model_path=args.yolo_model)
    if not detector.is_ready:
        print(f"YOLO model not ready: {detector.load_error}", file=sys.stderr)
        return 1

    if args.cnn_model is None:
        print("CNN_MODEL_PATH not set and --cnn-model not provided", file=sys.stderr)
        return 1
    reader = CnnDigitReader(model_path=args.cnn_model, strict=False)
    if not reader.is_ready:
        print(f"CNN model not ready: {reader.load_error}", file=sys.stderr)
        return 1

    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(args.src.glob(ext))

    manifest: dict[str, str] = {}  # warped_filename -> original_filename
    success = 0
    failed = []

    for img_path in sorted(images):
        image = cv2.imread(str(img_path))
        if image is None:
            failed.append((img_path.name, "unreadable"))
            continue

        det = detector.detect(image)
        if det is None:
            failed.append((img_path.name, "YOLO failed"))
            continue

        warped, corners, conf = det
        stem = img_path.stem

        # Save warped grid image.
        warped_name = f"{stem}_warped.jpg"
        cv2.imwrite(str(args.dst / warped_name), warped)

        # Save corners.
        corners_data = {
            "corners": corners.tolist(),
            "confidence": round(float(conf), 4),
            "image_width": image.shape[1],
            "image_height": image.shape[0],
        }
        (args.dst / f"{stem}_corners.json").write_text(
            json.dumps(corners_data, indent=2) + "\n"
        )

        # OCR pre-annotation + solve attempt.
        result = recognize(reader, warped)
        grid = result.grid
        givens = result.givens
        avg_conf = result.metadata.get("average_confidence")
        solved = result.solution is not None

        # Priority score: 0.0 (needs review) → 1.0 (confident + solvable).
        # Unsolvable → 0.0; solvable → scale by OCR confidence.
        priority_score = round(float(avg_conf or 0) if solved else 0.0, 4)

        ocr_data = {
            "grid": grid,
            "givens": givens,
            "confidence": avg_conf,
            "solved": solved,
            "priority_score": priority_score,
        }
        (args.dst / f"{stem}_ocr.json").write_text(
            json.dumps(ocr_data, indent=2) + "\n"
        )

        tag = "SOLVED" if solved else "FAILED"
        manifest[warped_name] = img_path.name
        print(f"  {tag:6s} {img_path.name:35s} conf={conf:.3f} givens={givens} score={priority_score}")
        success += 1

    # Write manifest.
    (args.dst / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )

    print(f"\nGenerated: {success}, Failed: {len(failed)}")
    for name, reason in failed:
        print(f"  SKIP {name}: {reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
