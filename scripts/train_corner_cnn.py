"""Train corner regression CNN for sudoku grid detection.

Trains on the reviewed corners in data/labels/sudoku_corners.csv and the source
images under data/raw/images/ (including the gitignored local/).

Usage:
    uv run --extra ml python scripts/train_corner_cnn.py [--version X.Y.Z]
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._paths import DATA_IMAGE_DIR, DEFAULT_CORNERS, load_corner_labels
from scripts._provenance import file_sha256, file_sha256_many, git_state

DEFAULT_LABELS = REPO_ROOT / "data" / "labels" / "sudoku_labels.json"


def _import_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        return torch, DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError("PyTorch required. Run: uv sync --extra ml") from exc


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

INPUT_SIZE = 256


def _letterbox(image: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Letterbox resize to INPUT_SIZE x INPUT_SIZE.

    Returns (padded_image, scale, pad_x, pad_y).
    """
    h, w = image.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (INPUT_SIZE - new_w) / 2.0
    pad_y = (INPUT_SIZE - new_h) / 2.0
    top = int(round(pad_y - 0.1))
    left = int(round(pad_x - 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, INPUT_SIZE - new_h - top,
        left, INPUT_SIZE - new_w - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return padded, scale, float(left), float(top)


def _corners_to_letterbox(
    corners: np.ndarray, w: int, h: int, scale: float, pad_x: float, pad_y: float
) -> np.ndarray:
    """Convert normalized [0,1] corners to letterbox-normalized [0,1] coords."""
    lb = np.zeros_like(corners)
    lb[:, 0] = (corners[:, 0] * w * scale + pad_x) / INPUT_SIZE
    lb[:, 1] = (corners[:, 1] * h * scale + pad_y) / INPUT_SIZE
    return lb.astype(np.float32)


def _apply_augmentation(
    image: np.ndarray, corners: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply geometric + photometric augmentation to image and corners.

    Args:
        image: (INPUT_SIZE, INPUT_SIZE, 3) uint8 letterboxed image
        corners: (4, 2) normalized [0,1] in letterbox space

    Returns:
        (augmented_image, augmented_corners)
    """
    h, w = image.shape[:2]
    pts = corners.copy() * np.array([w, h], dtype=np.float32)

    # Random perspective
    if random.random() < 0.5:
        max_d = 0.08 * max(w, h)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.random.uniform(-max_d, max_d, src.shape).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        image = cv2.warpPerspective(image, M, (w, h), borderValue=(114, 114, 114))
        pts_h = np.hstack([pts, np.ones((4, 1), dtype=np.float32)])
        pts_t = (M @ pts_h.T).T
        pts = pts_t[:, :2] / pts_t[:, 2:3]

    # Random affine (rotation + scale + translate)
    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        scale = random.uniform(0.8, 1.2)
        tx = random.uniform(-0.1, 0.1) * w
        ty = random.uniform(-0.1, 0.1) * h
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        image = cv2.warpAffine(image, M, (w, h), borderValue=(114, 114, 114))
        pts_h = np.hstack([pts, np.ones((4, 1), dtype=np.float32)])
        pts = (M @ pts_h.T).T

    # Horizontal flip
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        pts[:, 0] = w - pts[:, 0]
        # Swap TL<->TR, BL<->BR
        pts = pts[[1, 0, 3, 2]]

    # Clamp and re-normalize
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    corners_out = pts / np.array([w, h], dtype=np.float32)

    # Photometric augmentation (no label change)
    if random.random() < 0.3:
        alpha = random.uniform(0.7, 1.3)
        beta = random.uniform(-30, 30)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    if random.random() < 0.2:
        k = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (k, k), 0)

    if random.random() < 0.1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image, corners_out.astype(np.float32)


def load_dataset(corners_path: Path, images_dir: Path = DATA_IMAGE_DIR):
    """Return [(image_path, corners_4x2_normalized)]; every labeled image must exist."""
    paths = {p.name: p for p in images_dir.rglob("*") if p.is_file()}
    samples = []
    for name, corners in load_corner_labels(corners_path).items():
        if name not in paths:
            raise FileNotFoundError(f"Image for corner label not found under {images_dir}: {name}")
        samples.append((paths[name], corners))
    return samples


def split_dataset(samples, labels_path: Path, val_ratio: float, seed: int):
    """Split by puzzle, so copies of one grid (rescaled, re-photographed) stay together."""
    grids = {Path(name).name: json.dumps(grid) for name, grid in json.loads(labels_path.read_text()).items()}
    groups = sorted({grids.get(path.name, path.name) for path, _ in samples})
    random.Random(seed).shuffle(groups)
    val_groups = set(groups[: max(1, int(round(len(groups) * val_ratio)))])
    train = [s for s in samples if grids.get(s[0].name, s[0].name) not in val_groups]
    val = [s for s in samples if grids.get(s[0].name, s[0].name) in val_groups]
    return train, val


class CornerDataset:
    """PyTorch Dataset for corner regression."""

    def __init__(self, samples, augment: bool):
        torch, _, Dataset = _import_torch()

        class _Impl(Dataset):
            def __init__(self, inner_samples, use_augment):
                self.samples = inner_samples
                self.augment = use_augment

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                img_path, corners = self.samples[idx]
                image = cv2.imread(str(img_path))
                if image is None:
                    raise FileNotFoundError(f"Cannot read: {img_path}")

                h, w = image.shape[:2]
                padded, scale, pad_x, pad_y = _letterbox(image)
                lb_corners = _corners_to_letterbox(corners, w, h, scale, pad_x, pad_y)

                if self.augment:
                    padded, lb_corners = _apply_augmentation(padded, lb_corners)

                # HWC BGR → CHW RGB, float32 [0, 1]
                tensor = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
                tensor = torch.from_numpy(tensor.copy())
                target = torch.from_numpy(lb_corners.flatten())  # (8,)
                return tensor, target

        self.impl = _Impl(samples, augment)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compute corner_mae and warp_iou.

    Args:
        pred: (N, 8) predicted normalized corners
        target: (N, 8) ground truth normalized corners
    """
    # Corner MAE
    corner_mae = float(np.abs(pred - target).mean())

    # Warp IoU: for each sample, compute IoU of the quadrilateral
    ious = []
    for i in range(pred.shape[0]):
        p_pts = (pred[i].reshape(4, 2) * INPUT_SIZE).astype(np.int32)
        t_pts = (target[i].reshape(4, 2) * INPUT_SIZE).astype(np.int32)

        p_mask = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
        t_mask = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
        cv2.fillConvexPoly(p_mask, p_pts, 1)
        cv2.fillConvexPoly(t_mask, t_pts, 1)

        intersection = float((p_mask & t_mask).sum())
        union = float((p_mask | t_mask).sum())
        ious.append(intersection / union if union > 0 else 0.0)

    return {
        "corner_mae": corner_mae,
        "warp_iou": float(np.mean(ious)),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train corner regression CNN")
    parser.add_argument("--corners", type=Path, default=DEFAULT_CORNERS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="Groups the val split by puzzle")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-epochs", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--version", type=str, default="1.0.0")
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="Train with uncommitted changes; the diff is logged as git_diff.patch",
    )
    return parser.parse_args()


def main() -> int:
    import mlflow

    torch, DataLoader, _ = _import_torch()

    args = parse_args()
    git_tags, git_diff = git_state(args.allow_dirty)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = load_dataset(args.corners)
    train_samples, val_samples = split_dataset(samples, args.labels, args.val_ratio, args.seed)
    train_loader = DataLoader(
        CornerDataset(train_samples, augment=True).impl,
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        CornerDataset(val_samples, augment=False).impl,
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    from backend.cv.corner_cnn_arch import create_corner_model
    model = create_corner_model().to(device)

    criterion = torch.nn.SmoothL1Loss()

    # Phase 1: freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW([
        {"params": model.head.parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)

    mlflow.set_experiment("sudoku-corner-cnn")
    with mlflow.start_run() as run:
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "head_lr": args.head_lr,
            "backbone_lr": args.backbone_lr,
            "weight_decay": args.weight_decay,
            "freeze_epochs": args.freeze_epochs,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "patience": args.patience,
            "version": args.version,
            "num_train": len(train_samples),
            "num_val": len(val_samples),
            "input_size": INPUT_SIZE,
        })
        mlflow.set_tags({
            **git_tags,
            "corners_sha256": file_sha256(args.corners),
            "images_sha256": file_sha256_many(path for path, _ in samples),
        })
        if git_diff:
            mlflow.log_text(git_diff, "git_diff.patch")
        mlflow.log_dict({
            "train": sorted(path.name for path, _ in train_samples),
            "val": sorted(path.name for path, _ in val_samples),
        }, "split.json")

        best_warp_iou = -1.0
        best_state = None
        epochs_no_improve = 0
        global_step = 0

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()

            # Phase transition: unfreeze backbone after freeze_epochs
            if epoch == args.freeze_epochs + 1:
                for param in model.backbone.parameters():
                    param.requires_grad = True
                optimizer = torch.optim.AdamW([
                    {"params": model.backbone.parameters(), "lr": args.backbone_lr},
                    {"params": model.head.parameters(), "lr": args.head_lr / 10},
                ], weight_decay=args.weight_decay)
                print(f"=== Epoch {epoch}: unfreezing backbone ===")

            model.train()
            train_loss = 0.0
            for images, targets in train_loader:
                images = images.to(device)
                targets = targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                preds = model(images)
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                global_step += 1
                mlflow.log_metric("train_loss", loss.item(), step=global_step)

            train_loss /= max(1, len(train_loader))

            # Validation
            model.eval()
            all_preds, all_targets = [], []
            val_loss = 0.0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    targets = targets.to(device)
                    preds = model(images)
                    val_loss += criterion(preds, targets).item()
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

            val_loss /= max(1, len(val_loader))
            preds_np = np.concatenate(all_preds)
            targets_np = np.concatenate(all_targets)
            metrics = compute_metrics(preds_np, targets_np)

            current_lr = optimizer.param_groups[0]["lr"]
            epoch_duration = time.time() - epoch_start

            print(
                f"epoch={epoch:03d} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"corner_mae={metrics['corner_mae']:.4f} "
                f"warp_iou={metrics['warp_iou']:.4f}"
            )

            mlflow.log_metrics({
                "train_loss_epoch": train_loss,
                "val_loss": val_loss,
                "corner_mae": metrics["corner_mae"],
                "warp_iou": metrics["warp_iou"],
                "lr": current_lr,
                "epoch_duration_sec": epoch_duration,
            }, step=epoch)

            if metrics["warp_iou"] > best_warp_iou:
                best_warp_iou = metrics["warp_iou"]
                best_state = {
                    "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "version": args.version,
                    "input_size": INPUT_SIZE,
                    "best_warp_iou": best_warp_iou,
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= args.patience:
                print(f"Early stop at epoch={epoch}")
                break

        if best_state is None:
            raise RuntimeError("No valid checkpoint produced")

        with tempfile.TemporaryDirectory() as tmpdir:
            pt_path = Path(tmpdir) / f"sudoku_corner_cnn_v{args.version}.pt"
            torch.save(best_state, pt_path)
            mlflow.log_artifact(str(pt_path))

        print(f"Run ID: {run.info.run_id}")
        print(f"Best warp_iou: {best_warp_iou:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
