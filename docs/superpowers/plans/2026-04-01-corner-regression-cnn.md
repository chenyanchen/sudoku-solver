# Corner Regression CNN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace YOLO26n-pose with a lightweight MobileNetV3-Small corner regression CNN that directly outputs 4 sudoku grid corners from a photo.

**Architecture:** MobileNetV3-Small (ImageNet pretrained) backbone + Linear regression head → 8 normalized values (4 corners × 2 coords). Integrates with existing `GridDetector` protocol, MLflow tracking, Label Studio annotation loop, and release pipeline.

**Tech Stack:** PyTorch, timm (MobileNetV3), onnxruntime, MLflow, Label Studio

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `backend/cv/corner_cnn_arch.py` | Create | PyTorch model: MobileNetV3-Small + regression head |
| `backend/cv/corner_cnn_detector.py` | Create | ONNX inference, implements `GridDetector` protocol |
| `scripts/train_corner_cnn.py` | Create | Training script with MLflow, freeze/unfreeze, augmentation |
| `scripts/export_corner_cnn_onnx.py` | Create | Export .pt → .onnx |
| `scripts/migrate_yolo_to_corner.py` | Create | One-time: 16-kpt YOLO labels → 4-corner CSV |
| `data/corner_cnn/` | Create | Training data directory (labels.csv + images/) |
| `backend/api/routes.py` | Modify | Detector chain: CornerCnn → OpenCV |
| `ml_backend/app.py` | Modify | `/predict` 4 corners, `/fit` writes CSV, load CornerCnnDetector |
| `scripts/release_model.py` | Modify | Delete `release_yolo()`, add `release_corner_cnn()` |
| `scripts/labeling_pipeline.sh` | Modify | Train/release: corner-cnn replaces yolo |
| `.gitignore` | Modify | Add `data/corner_cnn/` |

---

### Task 1: Create model architecture

**Files:**
- Create: `backend/cv/corner_cnn_arch.py`

- [ ] **Step 1: Create `backend/cv/corner_cnn_arch.py`**

```python
"""MobileNetV3-Small based corner regression model for sudoku grid detection."""

from __future__ import annotations


def create_corner_model():
    """Create MobileNetV3-Small + regression head for 4-corner prediction.

    Output: 8 values in [0, 1] — (x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl).
    """
    try:
        import timm
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError(
            "timm and PyTorch required. Run: uv sync --extra ml"
        ) from exc

    backbone = timm.create_model(
        "mobilenetv3_small_100",
        pretrained=True,
        num_classes=0,  # remove classifier, return features
    )
    feature_dim = backbone.num_features  # 576 for mobilenetv3_small

    class CornerRegressionCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(64, 8),
                nn.Sigmoid(),
            )

        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)

    return CornerRegressionCNN()
```

- [ ] **Step 2: Verify model creation and output shape**

```bash
uv run python -c "
from backend.cv.corner_cnn_arch import create_corner_model
import torch
model = create_corner_model()
x = torch.randn(1, 3, 256, 256)
out = model(x)
print(f'Output shape: {out.shape}')  # (1, 8)
print(f'Output range: [{out.min():.3f}, {out.max():.3f}]')  # [0, 1]
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total params: {total:,}')
print(f'Trainable params: {trainable:,}')
"
```

Expected: Output shape (1, 8), values in [0, 1].

- [ ] **Step 3: Commit**

```bash
git add backend/cv/corner_cnn_arch.py
git commit -m "feat: add MobileNetV3 corner regression model architecture"
```

---

### Task 2: Create ONNX detector

**Files:**
- Create: `backend/cv/corner_cnn_detector.py`

- [ ] **Step 1: Create `backend/cv/corner_cnn_detector.py`**

```python
"""Corner regression CNN detector: locates 4 sudoku grid corners via ONNX."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .detector import DetectionResult

LOGGER = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError:
    ort = None

DEFAULT_CORNER_CNN_MODEL = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "releases"
    / "sudoku_corner_cnn.onnx"
)

INPUT_SIZE = 256


class CornerCnnDetector:
    """Detect sudoku grid corners using a lightweight regression CNN (ONNX).

    Returns 4 corner coordinates (TL, TR, BR, BL) in original image space.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
    ) -> None:
        self._session = None
        self._input_name = ""
        self._load_error: Optional[str] = None
        self.last_confidence: float = 0.0

        path = Path(model_path) if model_path else DEFAULT_CORNER_CNN_MODEL
        self._load_model(path)

    @property
    def is_ready(self) -> bool:
        return self._session is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _load_model(self, path: Path) -> None:
        if ort is None:
            self._load_error = "onnxruntime not installed"
            return
        if not path.exists():
            self._load_error = f"Corner CNN model not found: {path}"
            return
        try:
            self._session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            LOGGER.info("Corner CNN detector loaded: %s", path.name)
        except Exception as exc:
            self._load_error = str(exc)

    def detect(self, image: np.ndarray) -> DetectionResult | None:
        if not self.is_ready:
            return None

        h_orig, w_orig = image.shape[:2]
        blob, scale, pad_x, pad_y = self._preprocess(image)

        outputs = self._session.run(None, {self._input_name: blob})
        raw = outputs[0][0]  # shape: (8,)

        # raw is [x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl] in [0, 1]
        # relative to the 256x256 letterboxed input
        corners_norm = raw.reshape(4, 2)

        # Map from letterbox space back to original image coords
        corners = np.zeros((4, 2), dtype=np.float32)
        for i in range(4):
            lbx = corners_norm[i, 0] * INPUT_SIZE
            lby = corners_norm[i, 1] * INPUT_SIZE
            corners[i, 0] = max(0.0, min((lbx - pad_x) / scale, w_orig - 1))
            corners[i, 1] = max(0.0, min((lby - pad_y) / scale, h_orig - 1))

        self.last_confidence = 1.0  # regression model has no confidence score

        return DetectionResult(
            corners=corners,
            detector_name="corner_cnn",
        )

    def _preprocess(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, float, float, float]:
        """Letterbox resize to INPUT_SIZE x INPUT_SIZE, return (blob, scale, pad_x, pad_y)."""
        h, w = image.shape[:2]
        scale = INPUT_SIZE / max(h, w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_x = (INPUT_SIZE - new_w) / 2.0
        pad_y = (INPUT_SIZE - new_h) / 2.0
        top = int(round(pad_y - 0.1))
        bottom = int(round(pad_y + 0.1))
        left = int(round(pad_x - 0.1))
        right = int(round(pad_x + 0.1))
        padded = cv2.copyMakeBorder(
            resized, top, INPUT_SIZE - new_h - top,
            left, INPUT_SIZE - new_w - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )

        # HWC BGR → CHW RGB, float32 [0, 1]
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return blob[np.newaxis], scale, float(left), float(top)
```

- [ ] **Step 2: Commit**

```bash
git add backend/cv/corner_cnn_detector.py
git commit -m "feat: add CornerCnnDetector with ONNX inference"
```

---

### Task 3: Migrate YOLO data to corner CSV format

**Files:**
- Create: `scripts/migrate_yolo_to_corner.py`
- Create: `data/corner_cnn/` directory

- [ ] **Step 1: Create `scripts/migrate_yolo_to_corner.py`**

```python
"""One-time migration: convert 16-keypoint YOLO labels to 4-corner CSV.

Usage:
    uv run python scripts/migrate_yolo_to_corner.py
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
YOLO_LABELS = REPO_ROOT / "data" / "yolo_grid" / "labels" / "train"
YOLO_IMAGES = REPO_ROOT / "data" / "yolo_grid" / "images" / "train"
CORNER_DIR = REPO_ROOT / "data" / "corner_cnn"
CORNER_IMAGES = CORNER_DIR / "images"
CORNER_CSV = CORNER_DIR / "labels.csv"

# In 16-kpt YOLO format, keypoints are at indices 0-15 (row-major 4x4).
# Outer corners: P00=0 (TL), P03=3 (TR), P33=15 (BR), P30=12 (BL)
_TL, _TR, _BR, _BL = 0, 3, 15, 12


def main() -> int:
    CORNER_IMAGES.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []

    for label_path in sorted(YOLO_LABELS.glob("*.txt")):
        stem = label_path.stem
        # Find matching image
        image_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = YOLO_IMAGES / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            print(f"  SKIP {stem}: no matching image")
            continue

        # Parse YOLO label: class cx cy w h kx0 ky0 kx1 ky1 ... kx15 ky15
        parts = label_path.read_text().strip().split()
        if len(parts) != 37:
            print(f"  SKIP {stem}: expected 37 parts, got {len(parts)}")
            continue

        # Extract 16 keypoints (normalized [0, 1])
        kpts = []
        for i in range(16):
            kx = float(parts[5 + i * 2])
            ky = float(parts[5 + i * 2 + 1])
            kpts.append((kx, ky))

        tl = kpts[_TL]
        tr = kpts[_TR]
        br = kpts[_BR]
        bl = kpts[_BL]

        # Symlink image
        dst = CORNER_IMAGES / image_path.name
        if not dst.exists():
            dst.symlink_to(image_path.resolve())

        rows.append({
            "image": image_path.name,
            "x_tl": f"{tl[0]:.6f}", "y_tl": f"{tl[1]:.6f}",
            "x_tr": f"{tr[0]:.6f}", "y_tr": f"{tr[1]:.6f}",
            "x_br": f"{br[0]:.6f}", "y_br": f"{br[1]:.6f}",
            "x_bl": f"{bl[0]:.6f}", "y_bl": f"{bl[1]:.6f}",
        })

    # Write CSV
    fieldnames = ["image", "x_tl", "y_tl", "x_tr", "y_tr", "x_br", "y_br", "x_bl", "y_bl"]
    with CORNER_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Migrated {len(rows)} samples to {CORNER_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run migration**

```bash
uv run python scripts/migrate_yolo_to_corner.py
```

Expected: `Migrated 38 samples to data/corner_cnn/labels.csv`

- [ ] **Step 3: Add `data/corner_cnn/` to `.gitignore`**

Add under the existing derived data section:

```gitignore
# Derived data (regenerated by scripts)
data/warped_grids/
data/yolo_grid/
data/corner_cnn/
```

- [ ] **Step 4: Commit**

```bash
git add scripts/migrate_yolo_to_corner.py .gitignore
git commit -m "feat: add YOLO-to-corner data migration script"
```

---

### Task 4: Create training script

**Files:**
- Create: `scripts/train_corner_cnn.py`

This is the largest task. The training script includes:
- MLflow integration (params, step-level train_loss, epoch-level val metrics, .pt artifact)
- Two-phase freeze/unfreeze training
- Geometric augmentation with synchronized corner label transforms
- `corner_mae` and `warp_iou` evaluation metrics

- [ ] **Step 1: Create `scripts/train_corner_cnn.py`**

```python
"""Train corner regression CNN for sudoku grid detection.

Usage:
    uv run python scripts/train_corner_cnn.py [--epochs 60] [--lr 1e-3]
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _import_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        return torch, DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError("PyTorch required. Run: uv sync --extra ml") from exc


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


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


def load_dataset(csv_path: Path, images_dir: Path):
    """Load dataset from CSV. Returns list of (image_path, corners_4x2_normalized)."""
    samples = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = images_dir / row["image"]
            if not img_path.exists():
                continue
            corners = np.array([
                [float(row["x_tl"]), float(row["y_tl"])],
                [float(row["x_tr"]), float(row["y_tr"])],
                [float(row["x_br"]), float(row["y_br"])],
                [float(row["x_bl"]), float(row["y_bl"])],
            ], dtype=np.float32)
            samples.append((img_path, corners))
    return samples


def split_dataset(samples, val_ratio: float, seed: int):
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(indices) * val_ratio)))
    val_idx = set(indices[:val_count])
    train = [s for i, s in enumerate(samples) if i not in val_idx]
    val = [s for i, s in enumerate(samples) if i in val_idx]
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
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "corner_cnn")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--version", type=str, default="1.0.0")
    return parser.parse_args()


def main() -> int:
    import mlflow

    torch, DataLoader, _ = _import_torch()

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = args.data_dir / "labels.csv"
    images_dir = args.data_dir / "images"
    if not csv_path.exists():
        raise FileNotFoundError(f"Labels not found: {csv_path}")

    samples = load_dataset(csv_path, images_dir)
    if len(samples) < 5:
        raise RuntimeError(f"Too few samples ({len(samples)}). Need at least 5.")

    train_samples, val_samples = split_dataset(samples, args.val_ratio, args.seed)
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
        mlflow.set_tag("git_commit", _git_commit_hash())

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
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
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
```

- [ ] **Step 2: Test training with 3 epochs**

```bash
uv run python scripts/migrate_yolo_to_corner.py
export MLFLOW_TRACKING_URI="file://$(pwd)/mlruns"
uv run python scripts/train_corner_cnn.py --epochs 3 --version test
```

Expected: Training completes, prints Run ID and warp_iou.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_corner_cnn.py
git commit -m "feat: add corner CNN training script with MLflow integration"
```

---

### Task 5: Create ONNX export script

**Files:**
- Create: `scripts/export_corner_cnn_onnx.py`

- [ ] **Step 1: Create `scripts/export_corner_cnn_onnx.py`**

```python
"""Export corner regression CNN checkpoint to ONNX format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.cv.corner_cnn_arch import create_corner_model


def parse_args():
    parser = argparse.ArgumentParser(description="Export corner CNN to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True, help="PyTorch .pt path")
    parser.add_argument("--output", type=Path, required=True, help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    payload = torch.load(args.checkpoint, map_location="cpu")
    input_size = payload.get("input_size", 256)

    model = create_corner_model()
    model.load_state_dict(payload["state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, input_size, input_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        input_names=["input"],
        output_names=["corners"],
        dynamic_axes={"input": {0: "batch"}, "corners": {0: "batch"}},
        opset_version=args.opset,
    )

    print(f"Exported ONNX: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Commit**

```bash
git add scripts/export_corner_cnn_onnx.py
git commit -m "feat: add corner CNN ONNX export script"
```

---

### Task 6: Update release_model.py

**Files:**
- Modify: `scripts/release_model.py`

- [ ] **Step 1: Replace `release_yolo()` with `release_corner_cnn()`**

Delete the entire `release_yolo()` function (lines 123-164). Add in its place:

```python
def release_corner_cnn(run_id: str, release_version: str) -> int:
    client = MlflowClient()
    run = client.get_run(run_id)

    warp_iou = run.data.metrics.get("warp_iou", 0.0)

    print("Corner CNN Release")
    print(f"  Run ID:    {run_id}")
    print(f"  warp_iou:  {warp_iou:.4f}")
    print(f"  Version:   {release_version}")

    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts = client.list_artifacts(run_id)
        pt_artifacts = [a.path for a in artifacts if a.path.endswith(".pt")]
        if not pt_artifacts:
            raise SystemExit(f"No .pt artifact found in run {run_id}")

        local_path = client.download_artifacts(run_id, pt_artifacts[0], tmpdir)
        pt_path = Path(local_path)

        output_path = RELEASES_DIR / f"sudoku_corner_cnn_{release_version}.onnx"
        result = subprocess.run(
            [
                sys.executable, str(REPO_ROOT / "scripts" / "export_corner_cnn_onnx.py"),
                "--checkpoint", str(pt_path),
                "--output", str(output_path),
            ],
            cwd=str(REPO_ROOT),
        )
        if result.returncode != 0:
            raise SystemExit(f"ONNX export failed with code {result.returncode}")

        canonical = RELEASES_DIR / "sudoku_corner_cnn.onnx"
        canonical.unlink(missing_ok=True)
        canonical.symlink_to(output_path.name)

        _write_release_metadata(
            artifact_path=output_path,
            model_type="corner_cnn",
            model_version=release_version,
            run=run,
            metric_name="warp_iou",
            metric_value=warp_iou,
        )

    print(f"  Output:  {output_path}")
    return 0
```

- [ ] **Step 2: Update `main()` argument choices and dispatch**

Change `choices=["cnn", "yolo"]` to `choices=["cnn", "corner-cnn"]`:

```python
def main() -> int:
    parser = argparse.ArgumentParser(description="Release model from MLflow")
    parser.add_argument("model", choices=["cnn", "corner-cnn"], help="Model type to release")
    parser.add_argument(
        "--version",
        required=True,
        type=_validate_release_version,
        help="Unified release version in YYYYMMDD.N format",
    )
    parser.add_argument("--run-id", required=True, help="MLflow run ID to release")
    args = parser.parse_args()

    if args.model == "cnn":
        return release_cnn(args.run_id, args.version)
    else:
        return release_corner_cnn(args.run_id, args.version)
```

- [ ] **Step 3: Commit**

```bash
git add scripts/release_model.py
git commit -m "refactor: replace release_yolo with release_corner_cnn"
```

---

### Task 7: Update detector chain in routes.py

**Files:**
- Modify: `backend/api/routes.py`

- [ ] **Step 1: Replace YOLO with CornerCnn in `_get_detector()`**

Replace the YOLO import and initialization (lines 18, 61-69) with CornerCnn:

Change the import from:
```python
from ..cv.yolo_detector import YoloGridDetector
```
to:
```python
from ..cv.corner_cnn_detector import CornerCnnDetector
```

Replace the `_get_detector()` function body:

```python
def _get_detector() -> ChainDetector:
    """Lazy-init the detector chain: CornerCNN (preferred) → OpenCV (fallback)."""
    global _DETECTOR
    if _DETECTOR is not None:
        return _DETECTOR

    detectors = []

    corner_cnn = CornerCnnDetector(
        model_path=os.getenv("CORNER_CNN_MODEL_PATH"),
    )
    if corner_cnn.is_ready:
        detectors.append(corner_cnn)
        _LOGGER.info("Corner CNN grid detector ready")
    else:
        _LOGGER.info("Corner CNN unavailable: %s", corner_cnn.load_error)

    detectors.append(OpenCVGridDetector())
    _LOGGER.info("OpenCV grid detector ready (fallback)")

    _DETECTOR = ChainDetector(detectors)
    return _DETECTOR
```

- [ ] **Step 2: Commit**

```bash
git add backend/api/routes.py
git commit -m "refactor: detector chain uses CornerCnn instead of YOLO"
```

---

### Task 8: Update ML Backend

**Files:**
- Modify: `ml_backend/app.py`

This task adapts the ML backend for the new 4-corner Label Studio project.

- [ ] **Step 1: Replace YOLO loader with CornerCnn loader**

Replace `_get_yolo()` function and its `_yolo` global. Change the import and initialization to use `CornerCnnDetector`:

```python
_corner_cnn = None

def _get_corner_cnn():
    global _corner_cnn
    if _corner_cnn is None:
        from backend.cv.corner_cnn_detector import CornerCnnDetector
        _corner_cnn = CornerCnnDetector(
            model_path=os.getenv("CORNER_CNN_MODEL"),
        )
        if _corner_cnn.is_ready:
            logger.info("Corner CNN loaded: %s", Path(os.getenv("CORNER_CNN_MODEL", "")).name)
        else:
            logger.error("Corner CNN failed: %s", _corner_cnn.load_error)
    return _corner_cnn
```

Update `_validate_backend_model_env` to check `CORNER_CNN_MODEL` instead of `YOLO_MODEL`. Update `_current_release_versions` accordingly.

- [ ] **Step 2: Replace `_predict_keypoints()` for 4-corner project**

Replace the 16-keypoint prediction with 4-corner prediction:

```python
_CORNER_LABELS = ["TL", "TR", "BR", "BL"]


def _predict_corners(image: np.ndarray) -> tuple[list[dict], float]:
    detector = _get_corner_cnn()
    if not detector.is_ready:
        return [], 0.0
    det = detector.detect(image)
    if det is None:
        return [], 0.0

    h, w = image.shape[:2]
    corners = det.corners  # (4, 2) in pixel coords

    result = []
    for i, label in enumerate(_CORNER_LABELS):
        result.append({
            "from_name": "corners",
            "to_name": "image",
            "type": "keypointlabels",
            "value": {
                "x": round(float(corners[i, 0]) / w * 100, 4),
                "y": round(float(corners[i, 1]) / h * 100, 4),
                "width": 0.3,
                "keypointlabels": [label],
            },
            "score": round(detector.last_confidence, 4),
        })
    return result, detector.last_confidence
```

Update `_predict_ocr` to use `_get_corner_cnn()` instead of `_get_yolo()`.

- [ ] **Step 3: Replace `_fit_keypoints()` for 4-corner CSV**

Replace with a function that writes to `data/corner_cnn/labels.csv`:

```python
def _fit_corners(annotation: dict, task: dict) -> None:
    """Save corrected 4 corners to corner_cnn CSV format."""
    filename = _extract_filename(task)
    src_image = REPO_ROOT / "data" / "raw" / "images" / filename
    if not src_image.exists():
        logger.warning("fit/corners: source image not found: %s", filename)
        return

    image = cv2.imread(str(src_image))
    if image is None:
        return
    h, w = image.shape[:2]

    corner_map: dict[str, tuple[float, float]] = {}
    for r in annotation.get("result", []):
        if r.get("type") != "keypointlabels":
            continue
        labels = r["value"].get("keypointlabels", [])
        if not labels:
            continue
        x_norm = r["value"]["x"] / 100.0
        y_norm = r["value"]["y"] / 100.0
        corner_map[labels[0]] = (x_norm, y_norm)

    expected = ["TL", "TR", "BR", "BL"]
    missing = [c for c in expected if c not in corner_map]
    if missing:
        logger.warning("fit/corners: missing corners %s for %s", missing, filename)
        return

    # Copy image to corner_cnn dataset
    import shutil
    dst_dir = REPO_ROOT / "data" / "corner_cnn" / "images"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_img = dst_dir / Path(filename).name
    if not dst_img.exists():
        shutil.copy2(src_image, dst_img)

    # Append to CSV
    csv_path = REPO_ROOT / "data" / "corner_cnn" / "labels.csv"
    import csv
    fieldnames = ["image", "x_tl", "y_tl", "x_tr", "y_tr", "x_br", "y_br", "x_bl", "y_bl"]

    # Read existing to check for duplicates
    existing = set()
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing.add(row["image"])

    row_data = {
        "image": Path(filename).name,
        "x_tl": f"{corner_map['TL'][0]:.6f}",
        "y_tl": f"{corner_map['TL'][1]:.6f}",
        "x_tr": f"{corner_map['TR'][0]:.6f}",
        "y_tr": f"{corner_map['TR'][1]:.6f}",
        "x_br": f"{corner_map['BR'][0]:.6f}",
        "y_br": f"{corner_map['BR'][1]:.6f}",
        "x_bl": f"{corner_map['BL'][0]:.6f}",
        "y_bl": f"{corner_map['BL'][1]:.6f}",
    }

    if Path(filename).name in existing:
        # Update existing row
        rows = []
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["image"] == Path(filename).name:
                    rows.append(row_data)
                else:
                    rows.append(r)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        # Append new row
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row_data)

    logger.info("fit/corners: saved %s", filename)
```

Update the `/fit` endpoint to call `_fit_corners` instead of `_fit_keypoints` for the corners project, and update `_is_keypoints_project` detection logic.

- [ ] **Step 4: Commit**

```bash
git add ml_backend/app.py
git commit -m "refactor: ML backend uses CornerCnn for predict/fit"
```

---

### Task 9: Update labeling_pipeline.sh

**Files:**
- Modify: `scripts/labeling_pipeline.sh`

- [ ] **Step 1: Replace YOLO with corner-cnn in train command**

Replace the `train)` case:

```bash
  train)
    export MLFLOW_TRACKING_URI="file://${REPO_ROOT}/mlruns"

    echo "=== Training Corner CNN ==="
    uv run python scripts/train_corner_cnn.py

    echo ""
    echo "=== Training CNN OCR ==="
    uv run python scripts/train_cnn_ocr.py \
      --labels data/labels/sudoku_labels.json \
      --use-yolo --version 3.4.0
    ;;
```

Note: CNN OCR's `--use-yolo` flag still references the YOLO/CornerCnn detector for grid extraction during training data preparation. This flag name is misleading now but functionally it means "use the ML-based detector instead of OpenCV". Renaming is out of scope for this task.

- [ ] **Step 2: Update backend command to use CORNER_CNN_MODEL**

Replace the `backend)` case:

```bash
  backend)
    : "${CNN_MODEL_PATH:?CNN_MODEL_PATH must point to an explicit released CNN ONNX}"
    : "${CORNER_CNN_MODEL:?CORNER_CNN_MODEL must point to an explicit released corner CNN ONNX}"
    echo "=== Starting ML Backend on port ${2:-9090} ==="
    uv run python -m ml_backend --port "${2:-9090}"
    ;;
```

- [ ] **Step 3: Update help text**

```bash
  help|*)
    cat <<USAGE
Usage: $0 <command>

Commands:
  backend [port]  Start ML Backend server (default: 9090)
  generate        Generate warped grids for OCR project import
  train           Retrain Corner CNN + OCR CNN models (metrics logged to MLflow)
  release <cnn|corner-cnn> --version <YYYYMMDD.N> --run-id <id>  Export versioned model

Workflow:
  1. Release explicit models:
     $0 release corner-cnn --version 20260401.1 --run-id <id>
     $0 release cnn --version 20260401.1 --run-id <id>
  2. Start ML Backend:
     CNN_MODEL_PATH=models/releases/sudoku_digit_cnn.onnx \\
     CORNER_CNN_MODEL=models/releases/sudoku_corner_cnn.onnx \\
     $0 backend
  3. Connect in Label Studio: Settings → Model → http://localhost:9090
  4. Import images, label 4 corners (TL, TR, BR, BL)
  5. Train: $0 train
  6. Review metrics: mlflow ui --port 5000 --backend-store-uri file://\$(pwd)/mlruns

Environment variables:
  CNN_MODEL_PATH      (required for backend)
  CORNER_CNN_MODEL    (required for backend)
  MLFLOW_TRACKING_URI (default: file://\$REPO_ROOT/mlruns)
USAGE
    ;;
```

- [ ] **Step 4: Commit**

```bash
git add scripts/labeling_pipeline.sh
git commit -m "refactor: pipeline uses corner-cnn instead of yolo"
```

---

### Task 10: Delete YOLO training script

**Files:**
- Delete: `scripts/train_yolo_grid.py`

- [ ] **Step 1: Delete the file**

```bash
git rm scripts/train_yolo_grid.py
```

- [ ] **Step 2: Commit**

```bash
git commit -m "chore: remove YOLO training script (replaced by corner CNN)"
```
