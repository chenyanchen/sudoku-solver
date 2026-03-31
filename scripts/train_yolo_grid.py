"""Train YOLO26n-pose model for sudoku grid detection with 4-corner keypoints.

Usage:
    uv sync --extra ml
    uv run python scripts/train_yolo_grid.py [--epochs 50] [--model yolo26n-pose]
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def _write_split_yaml(split_root: Path) -> Path:
    yaml_path = split_root / "sudoku_grid.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {split_root}",
                "train: images/train",
                "val: images/val",
                "",
                "names:",
                "  0: sudoku_grid",
                "",
                "kpt_shape: [16, 2]",
                "flip_idx: []",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def prepare_dataset_split(
    source_root: Path,
    split_root: Path,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Path:
    src_images = source_root / "images" / "train"
    src_labels = source_root / "labels" / "train"
    if not src_images.exists() or not src_labels.exists():
        raise FileNotFoundError(f"YOLO dataset not found under {source_root}")

    raw_stems: list[str] = []
    aug_stems: list[str] = []
    for image_path in sorted(src_images.iterdir()):
        if not image_path.is_file():
            continue
        stem = image_path.stem
        label_path = src_labels / f"{stem}.txt"
        if not label_path.exists():
            continue
        if stem.startswith("aug_"):
            aug_stems.append(stem)
        else:
            raw_stems.append(stem)

    if not raw_stems:
        raise RuntimeError("YOLO dataset has no raw labeled samples")

    rng = random.Random(seed)
    shuffled = list(raw_stems)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 1
    val_stems = set(shuffled[:val_count])
    train_stems = [stem for stem in raw_stems if stem not in val_stems] + aug_stems

    for stem in train_stems:
        image_path = next(src_images.glob(f"{stem}.*"))
        _link_or_copy(image_path, split_root / "images" / "train" / image_path.name)
        _link_or_copy(src_labels / f"{stem}.txt", split_root / "labels" / "train" / f"{stem}.txt")

    for stem in sorted(val_stems):
        image_path = next(src_images.glob(f"{stem}.*"))
        _link_or_copy(image_path, split_root / "images" / "val" / image_path.name)
        _link_or_copy(src_labels / f"{stem}.txt", split_root / "labels" / "val" / f"{stem}.txt")

    return _write_split_yaml(split_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train YOLO grid detector")
    parser.add_argument("--model", default="yolo26n-pose.pt", help="Pretrained model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="Device (cpu, mps, cuda)")
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "yolo_grid" / "sudoku_grid.yaml",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--project",
        type=Path,
        default=REPO_ROOT / "models" / "yolo_runs",
    )
    parser.add_argument("--name", default="grid_detect")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. Run: uv add ultralytics")
        return 1

    model = YOLO(args.model)
    dataset_root = args.data.resolve().parent
    with tempfile.TemporaryDirectory(prefix="yolo-split-") as tmpdir:
        split_root = Path(tmpdir)
        split_yaml = prepare_dataset_split(
            source_root=dataset_root,
            split_root=split_root,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        data_path = str(split_yaml.resolve())

        os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "sudoku-yolo-grid")

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
            flipud=0.0,
            fliplr=0.0,
            mosaic=0.5,
            mixup=0.0,
        )

    # Tag the MLflow run with git commit (run is already ended by ultralytics)
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        experiment = client.get_experiment_by_name("sudoku-yolo-grid")
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs:
                git_hash = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
                ).strip()
                client.set_tag(runs[0].info.run_id, "git_commit", git_hash)
    except Exception:
        pass  # MLflow not configured or git not available

    return 0


def _has_mps() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
