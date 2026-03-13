"""Train YOLO26n-pose model for sudoku grid detection with 4-corner keypoints.

Usage:
    uv sync --extra ml
    uv run python scripts/train_yolo_grid.py [--epochs 100] [--model yolo26n-pose]
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLO grid detector")
    parser.add_argument("--model", default="yolo26n-pose.pt", help="Pretrained model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="Device (cpu, mps, cuda)")
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "yolo_grid" / "sudoku_grid.yaml",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=REPO_ROOT / "models" / "yolo_runs",
    )
    parser.add_argument("--name", default="grid_detect")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. Run: uv add ultralytics")
        return 1

    model = YOLO(args.model)

    # Resolve absolute path for YAML — ultralytics resolves `path: .`
    # relative to the YAML file location only when path is absolute.
    data_path = str(args.data.resolve())

    model.train(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or ("mps" if _has_mps() else "cpu"),
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        # Augmentation tuned for document/screen detection
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        perspective=0.0005,
        flipud=0.0,  # sudoku grids aren't flipped vertically
        fliplr=0.0,  # or horizontally (digits would be mirrored)
        mosaic=0.5,
        mixup=0.0,
    )

    # Export best model to ONNX
    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    if best_pt.exists():
        best_model = YOLO(str(best_pt))
        onnx_path = best_model.export(format="onnx", imgsz=args.imgsz, simplify=True)
        print(f"\nExported ONNX: {onnx_path}")

        # Copy to releases
        releases_dir = REPO_ROOT / "models" / "releases"
        releases_dir.mkdir(exist_ok=True)
        import shutil

        dst = releases_dir / "yolo_grid_detect.onnx"
        shutil.copy2(onnx_path, dst)
        print(f"Copied to: {dst}")

    return 0


def _has_mps() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
