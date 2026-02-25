"""Train Sudoku cell CNN classifier (blank + digits 1..9)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.api.routes import detect_grid_image, prepare_grid_for_ocr
from backend.cv.cell_extractor import clean_cell, extract_cells


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


def build_samples(labels_path: Path, input_size: int) -> list[CellSample]:
    with labels_path.open("r", encoding="utf-8") as f:
        labels_data = json.load(f)

    samples: list[CellSample] = []

    for image_name, grid in labels_data.items():
        image_path = (REPO_ROOT / image_name).resolve()
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

        detected = detect_grid_image(image)
        if detected is None:
            raise ValueError(f"Could not detect Sudoku grid in image: {image_path}")

        ocr_grid = prepare_grid_for_ocr(detected)
        cells = extract_cells(ocr_grid)
        if len(cells) != 81:
            raise ValueError(f"Expected 81 cells from {image_name}, got {len(cells)}")

        for row in range(9):
            for col in range(9):
                label = int(grid[row][col])
                cell = cells[row * 9 + col]
                cleaned = clean_cell(cell, cell_size=input_size)
                samples.append(CellSample(image=cleaned, label=label))

    return samples


def stratified_split(
    samples: list[CellSample], val_ratio: float, seed: int
) -> tuple[list[CellSample], list[CellSample]]:
    random.seed(seed)

    grouped: dict[int, list[CellSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.label, []).append(sample)

    train: list[CellSample] = []
    val: list[CellSample] = []

    for label, group in grouped.items():
        random.shuffle(group)
        val_count = max(1, int(len(group) * val_ratio))
        val.extend(group[:val_count])
        train.extend(group[val_count:])

    random.shuffle(train)
    random.shuffle(val)
    return train, val


def create_model(arch: str, num_classes: int):
    torch, nn, _, _ = _import_torch()

    if arch == "timm_mobilenetv3":
        try:
            import timm

            model = timm.create_model(
                "mobilenetv3_small_100",
                pretrained=False,
                in_chans=1,
                num_classes=num_classes,
            )
            return model
        except Exception:
            print("[WARN] timm unavailable, fallback to custom_small_cnn")

    class SudokuDigitCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 6 * 6, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    return SudokuDigitCNN()


def compute_metrics(logits, labels):
    torch, _, _, _ = _import_torch()

    preds = torch.argmax(logits, dim=1)
    acc = (preds == labels).float().mean().item()

    # Digit-only macro F1 for labels 1..9
    f1_scores = []
    for cls in range(1, 10):
        tp = ((preds == cls) & (labels == cls)).sum().item()
        fp = ((preds == cls) & (labels != cls)).sum().item()
        fn = ((preds != cls) & (labels == cls)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)

    macro_f1_digits = float(np.mean(f1_scores)) if f1_scores else 0.0

    blank_tp = ((preds == 0) & (labels == 0)).sum().item()
    blank_fp = ((preds == 0) & (labels != 0)).sum().item()
    blank_precision = blank_tp / (blank_tp + blank_fp + 1e-8)

    return {
        "accuracy": acc,
        "macro_f1_digits": macro_f1_digits,
        "blank_precision": float(blank_precision),
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
        default=REPO_ROOT / "data" / "sudoku_labels.json",
        help="JSON path: image -> 9x9 grid labels",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "models" / "checkpoints" / "sudoku_digit_cnn.pt",
        help="Output checkpoint path",
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
        choices=["custom_small_cnn", "timm_mobilenetv3"],
        default="custom_small_cnn",
    )
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--version", type=str, default="1.0.0")
    return parser.parse_args()


def main() -> int:
    torch, _, DataLoader, _ = _import_torch()

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = build_samples(args.labels, args.input_size)
    if len(samples) < 50:
        raise RuntimeError(
            "Training dataset is too small. Add more labeled Sudoku images first."
        )

    train_samples, val_samples = stratified_split(samples, args.val_ratio, args.seed)
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
        weight=torch.from_numpy(class_weights).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_score = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
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

        train_loss /= max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader, device)

        score = (
            0.65 * val_metrics["macro_f1_digits"]
            + 0.20 * val_metrics["blank_precision"]
            + 0.15 * val_metrics["accuracy"]
        )

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1_digits={val_metrics['macro_f1_digits']:.4f} "
            f"val_blank_precision={val_metrics['blank_precision']:.4f} "
            f"score={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_state = {
                "state_dict": model.state_dict(),
                "class_values": list(range(10)),
                "input_size": args.input_size,
                "version": args.version,
                "arch": args.arch,
                "best_score": best_score,
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"Early stop triggered at epoch={epoch}")
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint produced")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output)

    meta = {
        "version": args.version,
        "class_values": list(range(10)),
        "input_size": args.input_size,
        "arch": args.arch,
        "best_score": best_score,
    }
    meta_path = args.output.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved checkpoint: {args.output}")
    print(f"Saved metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
