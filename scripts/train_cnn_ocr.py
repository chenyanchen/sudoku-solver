"""Train Sudoku cell CNN classifier (blank + digits 1..9)."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._paths import DEFAULT_CORNERS, load_corner_labels, resolve_image_path
from scripts._provenance import file_sha256, git_state

from backend.ocr.pipeline import ocr_cells, prepare_grid_for_ocr
from backend.cv.cell_extractor import clean_cell
from backend.cv.detector import warp_grid
from backend.ocr.model_arch import create_model

import mlflow


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset

        return torch, nn, DataLoader, Dataset
    except Exception as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "PyTorch is required. Install ML stack first: uv sync --extra ml"
        ) from exc


@dataclass
class CellSample:
    image: np.ndarray
    label: int
    source: str  # Same givens share a split, including duplicate source images.


class SudokuCellDataset:
    """Torch dataset wrapper for Sudoku cell classification."""

    def __init__(self, samples: list[CellSample], augment: bool):
        torch, _, _, Dataset = _import_torch()

        class _DatasetImpl(Dataset):
            def __init__(self, inner_samples: list[CellSample], use_augment: bool):
                self.samples = inner_samples
                self.augment = use_augment

            def __len__(self) -> int:
                return len(self.samples)

            def __getitem__(self, idx: int):
                sample = self.samples[idx]
                image = sample.image.copy()
                label = int(sample.label)

                if self.augment:
                    image = apply_augmentations(image)

                image = image.astype(np.float32) / 255.0
                tensor = torch.from_numpy(image).unsqueeze(0)
                return tensor, label

        self.impl = _DatasetImpl(samples, augment)


def apply_augmentations(image: np.ndarray) -> np.ndarray:
    """Apply lightweight CV augmentations for robustness."""
    out = image.copy()

    if random.random() < 0.1:
        out = draw_highlight_ring(out)

    if random.random() < 0.35:
        alpha = random.uniform(0.85, 1.2)  # contrast
        beta = random.uniform(-18, 18)  # brightness
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    if random.random() < 0.25:
        k = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), sigmaX=0)

    if random.random() < 0.2:
        # Simulate anti-aliased thin strokes
        out = cv2.erode(out, np.ones((2, 2), np.uint8), iterations=1)

    if random.random() < 0.2:
        # Simulate thick rendering
        out = cv2.dilate(out, np.ones((2, 2), np.uint8), iterations=1)

    if random.random() < 0.15:
        noise = np.random.normal(0, 6, out.shape).astype(np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return out


def draw_highlight_ring(image: np.ndarray) -> np.ndarray:
    """Circle the cell like a game's highlight of the selected digit.

    Such rings survive binarization and made a circled 1 read as 9 or 6.
    """
    size = image.shape[0]
    center = (round(size / 2 + random.uniform(-2, 2)), round(size / 2 + random.uniform(-2, 2)))
    radius = round(size * random.uniform(0.38, 0.49))
    out = image.copy()
    cv2.circle(out, center, radius, random.randint(0, 60), random.randint(2, 4), cv2.LINE_AA)
    return out


def build_samples(
    labels_path: Path,
    input_size: int,
    corners_path: Path = DEFAULT_CORNERS,
) -> list[CellSample]:
    with labels_path.open("r", encoding="utf-8") as f:
        labels_data = json.load(f)

    corners_by_name = load_corner_labels(corners_path)

    samples: list[CellSample] = []

    for image_name, grid in labels_data.items():
        image_path = resolve_image_path(image_name, labels_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

        corners = corners_by_name.get(image_path.name)
        if corners is None:
            raise ValueError(f"Missing ground-truth corners for {image_name}")
        h, w = image.shape[:2]
        warped = warp_grid(image, corners * np.array([w, h]))
        source = json.dumps(grid)

        cells, _ = ocr_cells(prepare_grid_for_ocr(warped))

        for row in range(9):
            for col in range(9):
                label = int(grid[row][col])
                cell = cells[row * 9 + col]
                cleaned = clean_cell(cell, cell_size=input_size)
                samples.append(CellSample(image=cleaned, label=label, source=source))

    return samples


def grouped_split(
    samples: list[CellSample], val_ratio: float, seed: int
) -> tuple[list[CellSample], list[CellSample]]:
    """Keep every cell of a puzzle (including duplicate images) in one split."""
    sources = sorted({sample.source for sample in samples})
    if not 0 < val_ratio < 1 or len(sources) < 2:
        raise ValueError("Need at least two puzzles and 0 < val_ratio < 1")
    random.Random(seed).shuffle(sources)
    val_count = min(len(sources) - 1, max(1, round(len(sources) * val_ratio)))
    val_sources = set(sources[:val_count])
    return (
        [sample for sample in samples if sample.source not in val_sources],
        [sample for sample in samples if sample.source in val_sources],
    )


def compute_metrics(logits, labels):
    torch, _, _, _ = _import_torch()

    preds = torch.argmax(logits, dim=1)
    acc = (preds == labels).float().mean().item()

    # Digit-only macro F1 for labels 1..9
    f1_scores = []
    per_digit_f1: dict[str, float] = {}
    for cls in range(1, 10):
        tp = ((preds == cls) & (labels == cls)).sum().item()
        fp = ((preds == cls) & (labels != cls)).sum().item()
        fn = ((preds != cls) & (labels == cls)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)
        per_digit_f1[str(cls)] = float(f1)

    macro_f1_digits = float(np.mean(f1_scores)) if f1_scores else 0.0

    blank_tp = ((preds == 0) & (labels == 0)).sum().item()
    blank_fp = ((preds == 0) & (labels != 0)).sum().item()
    blank_precision = blank_tp / (blank_tp + blank_fp + 1e-8)

    # Confusion matrix (10×10): confusion[true][pred] = count.
    num_classes = 10
    confusion: list[list[int]] = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(labels.tolist(), preds.tolist()):
        confusion[int(t)][int(p)] += 1

    return {
        "accuracy": acc,
        "macro_f1_digits": macro_f1_digits,
        "blank_precision": float(blank_precision),
        "per_digit_f1": per_digit_f1,
        "confusion": confusion,
    }


def class_weights_from_samples(samples: list[CellSample]) -> np.ndarray:
    counts = np.zeros(10, dtype=np.float32)
    for sample in samples:
        counts[sample.label] += 1.0

    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    weights = inv / inv.sum() * len(inv)
    return weights.astype(np.float32)


def evaluate(model, data_loader, device):
    torch, _, _, _ = _import_torch()

    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item())
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(logits, labels)
    metrics["loss"] = total_loss / max(1, len(data_loader))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Sudoku OCR CNN model")
    parser.add_argument(
        "--labels",
        type=Path,
        default=REPO_ROOT / "data" / "labels" / "sudoku_labels.json",
        help="JSON path: image -> 9x9 grid labels",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-size", type=int, default=48)
    parser.add_argument(
        "--arch",
        choices=["custom_small_cnn", "custom_cnn_v3", "timm_mobilenetv3"],
        default="custom_cnn_v3",
    )
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--version", type=str, default="1.0.0")
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="Train with uncommitted changes; the diff is logged as git_diff.patch",
    )
    parser.add_argument(
        "--corners", type=Path, default=DEFAULT_CORNERS,
        help="Reviewed corner CSV; never train on predicted crops",
    )
    return parser.parse_args()


def main() -> int:
    torch, _, DataLoader, _ = _import_torch()

    args = parse_args()
    git_tags, git_diff = git_state(args.allow_dirty)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = build_samples(
        args.labels, args.input_size,
        corners_path=args.corners,
    )
    if len(samples) < 50:
        raise RuntimeError(
            "Training dataset is too small. Add more labeled Sudoku images first."
        )

    train_samples, val_samples = grouped_split(samples, args.val_ratio, args.seed)
    train_dataset = SudokuCellDataset(train_samples, augment=True).impl
    val_dataset = SudokuCellDataset(val_samples, augment=False).impl

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    model = create_model(args.arch, num_classes=10).to(device)

    class_weights = class_weights_from_samples(train_samples)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.from_numpy(class_weights).to(device),
        label_smoothing=0.1,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    mlflow.set_experiment("sudoku-cnn-ocr")
    with mlflow.start_run() as run:
        mlflow.log_params({
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "arch": args.arch,
            "version": args.version,
            "weight_decay": args.weight_decay,
            "label_smoothing": 0.1,
            "early_stopping_patience": args.patience,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "input_size": args.input_size,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "num_train_samples": len(train_samples),
            "num_val_samples": len(val_samples),
            "crop_source": "ground_truth",
            "split": "puzzle_group",
            "corners": str(args.corners),
        })
        mlflow.set_tags({
            **git_tags,
            "labels_sha256": file_sha256(args.labels),
            "corners_sha256": file_sha256(args.corners),
        })
        if git_diff:
            mlflow.log_text(git_diff, "git_diff.patch")
        labels_by_image = json.loads(args.labels.read_text())
        val_sources = {sample.source for sample in val_samples}
        mlflow.log_dict({
            "train": [name for name, grid in labels_by_image.items() if json.dumps(grid) not in val_sources],
            "val": [name for name, grid in labels_by_image.items() if json.dumps(grid) in val_sources],
        }, "split.json")

        best_score = -1.0
        best_state = None
        best_confusion: list[list[int]] = []
        epochs_no_improve = 0
        global_step = 0

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            model.train()
            train_loss = 0.0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += float(loss.item())
                global_step += 1
                mlflow.log_metric("train_loss", loss.item(), step=global_step)

            train_loss /= max(1, len(train_loader))
            val_metrics = evaluate(model, val_loader, device)

            score = (
                0.65 * val_metrics["macro_f1_digits"]
                + 0.20 * val_metrics["blank_precision"]
                + 0.15 * val_metrics["accuracy"]
            )

            # The small val split saturates at 1.0 early; on ties keep the later,
            # lower-LR checkpoint instead of freezing on the first perfect epoch.
            is_best = score >= best_score

            print(
                f"epoch={epoch:03d} train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1_digits={val_metrics['macro_f1_digits']:.4f} "
                f"val_blank_precision={val_metrics['blank_precision']:.4f} "
                f"score={score:.4f}"
            )

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            epoch_duration = time.time() - epoch_start

            epoch_metrics = {
                "train_loss_epoch": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1_digits"],
                "val_blank_precision": val_metrics["blank_precision"],
                "score": score,
                "lr": current_lr,
                "epoch_duration_sec": epoch_duration,
            }
            for digit, f1_val in val_metrics.get("per_digit_f1", {}).items():
                epoch_metrics[f"f1_digit_{digit}"] = f1_val
            mlflow.log_metrics(epoch_metrics, step=epoch)

            if is_best:
                best_score = score
                best_state = {
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "class_values": list(range(10)),
                    "input_size": args.input_size,
                    "version": args.version,
                    "arch": args.arch,
                    "best_score": best_score,
                }
                best_confusion = val_metrics.get("confusion", [])
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= args.patience:
                print(f"Early stop triggered at epoch={epoch}")
                break

        if best_state is None:
            raise RuntimeError("No valid checkpoint produced")

        with tempfile.TemporaryDirectory() as tmpdir:
            pt_path = Path(tmpdir) / f"sudoku_digit_cnn_v{args.version}.pt"
            torch.save(best_state, pt_path)
            mlflow.log_artifact(str(pt_path))

            # Save confusion matrix as PNG artifact
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 8))
                confusion = best_confusion
                im = ax.imshow(confusion, cmap="Blues")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_title("Confusion Matrix (best epoch)")
                ax.set_xticks(range(10))
                ax.set_yticks(range(10))
                cm_max = max(max(row) for row in confusion) if confusion else 1
                for i in range(10):
                    for j in range(10):
                        ax.text(j, i, str(confusion[i][j]), ha="center", va="center",
                                color="white" if confusion[i][j] > cm_max / 2 else "black",
                                fontsize=8)
                fig.colorbar(im)
                fig.tight_layout()
                cm_path = Path(tmpdir) / "confusion_matrix.png"
                fig.savefig(cm_path, dpi=100)
                plt.close(fig)
                mlflow.log_artifact(str(cm_path))
            except ImportError:
                pass  # matplotlib not available, skip confusion matrix image

        print(f"Run ID: {run.info.run_id}")
        print(f"Best score: {best_score:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
