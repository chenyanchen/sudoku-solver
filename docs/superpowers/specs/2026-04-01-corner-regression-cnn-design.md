# Corner Regression CNN Design

## Goal

Replace YOLO26n-pose with a lightweight corner regression CNN for sudoku grid detection from photos. The model directly regresses 4 corner coordinates (TL, TR, BR, BL) from an input image, producing a `DetectionResult` compatible with the existing pipeline.

## Why Not YOLO

YOLO26n-pose (2.7M params) is a general-purpose multi-class object detector with pose estimation. For our task (locate a single known geometric structure), it is over-parameterized and requires 500-1000+ training images. With 38 labeled images, it cannot generalize.

A direct regression CNN (~110K trainable params with frozen backbone) is fundamentally more data-efficient because:
- No detection pipeline overhead (anchors, NMS, class scores)
- Loss function directly optimizes corner accuracy
- ImageNet-pretrained backbone provides edge/texture priors for free

## Model Architecture

```
Input: (3, 256, 256) RGB
  → MobileNetV3-Small backbone (ImageNet pretrained)
  → AdaptiveAvgPool2d(1) → Flatten           # 576-dim
  → Linear(576, 64) → ReLU → Dropout(0.2)
  → Linear(64, 8) → Sigmoid                  # [x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl]
```

Output: 8 values in [0, 1], normalized coordinates. Corner order: TL, TR, BR, BL (matches `warp_grid()`).

| Property | Value |
|---|---|
| Backbone | MobileNetV3-Small (~1M params, ImageNet pretrained) |
| Input | RGB 256x256 (letterbox, preserve aspect ratio) |
| Output | 8 values (4 corners x 2 coords), Sigmoid-constrained to [0,1] |
| Loss | SmoothL1Loss on 8 normalized coordinates |
| Total params | ~1.05M |
| Trainable (frozen backbone) | ~40K |

### Why RGB, not grayscale

Grayscale saves negligible parameters (~300 out of 1M) but prevents clean use of ImageNet pretrained weights. With 38 images, pretrained weight integrity matters more.

## Training Strategy

### Two-phase learning rate

| Phase | Epochs | Backbone | Head LR | Backbone LR |
|---|---|---|---|---|
| 1 (warmup) | 1-5 | Frozen | 1e-3 | 0 |
| 2 (fine-tune) | 6-60 | Unfrozen | 1e-4 | 1e-5 |

Scheduler: CosineAnnealingLR. Early stopping: patience=15. Optimizer: AdamW.

### Data augmentation

Geometric transforms (must synchronously transform corner labels):

- `RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.7, 1.3))`
- `RandomPerspective(distortion_scale=0.2)`
- `HorizontalFlip(p=0.5)` — swaps TL↔TR, BL↔BR

Photometric transforms (no label change needed):

- `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)`
- `GaussianBlur(kernel_size=5, p=0.2)`
- `RandomGrayscale(p=0.1)`

### Train/Val split

80/20 stratified by image type (photos vs synthetic). Fixed seed for reproducibility.

### Evaluation metrics

| Metric | Definition | Target |
|---|---|---|
| `corner_mae` | Mean absolute error of 4 corners (normalized coords) | < 0.02 |
| `warp_iou` | IoU between predicted warp region and GT warp region | > 0.95 |

`warp_iou` is the primary business metric — a few pixels of corner offset is fine if the warped 450x450 grid is correct.

## Training Data

### Format

CSV file at `data/corner_cnn/labels.csv`:

```csv
image,x_tl,y_tl,x_tr,y_tr,x_br,y_br,x_bl,y_bl
IMG_8710.jpg,0.1717,0.3738,0.5117,0.4071,0.5178,0.6698,0.1711,0.7130
```

All coordinates normalized to [0, 1]. Images stored in `data/corner_cnn/images/`.

### Migration from YOLO data

One-time script extracts 4 outer corners from existing 16-keypoint labels:

- P00 → TL, P03 → TR, P33 → BR, P30 → BL

Symlinks images to `data/corner_cnn/images/`. Old YOLO data remains untouched.

### Directory structure

```
data/corner_cnn/
  labels.csv
  images/
    IMG_8710.jpg
    ...
```

## Infrastructure Integration

### MLflow

New experiment: `sudoku-corner-cnn`. Same pattern as CNN OCR:
- Step-level `train_loss`, epoch-level `val_corner_mae`, `val_warp_iou`
- Params, git commit tag, .pt artifact upload

### Label Studio

New Project with 4 keypoint labels: `TL`, `TR`, `BR`, `BL`.

Webhook → `/fit` → parse 4 corners → append to `labels.csv` + copy image to `data/corner_cnn/images/`.

`/predict` → run CornerCnnDetector → return 4 keypointlabels to Label Studio.

### Pipeline commands

```bash
# Train
./scripts/labeling_pipeline.sh train    # trains corner-cnn + cnn-ocr

# Release
./scripts/labeling_pipeline.sh release corner-cnn
./scripts/labeling_pipeline.sh release cnn
```

## Detector Integration

### CornerCnnDetector

New file: `backend/cv/corner_cnn_detector.py`. Implements `GridDetector` protocol.

```python
class CornerCnnDetector:
    def detect(self, image: np.ndarray) -> DetectionResult | None:
        # 1. Letterbox resize to 256x256
        # 2. ONNX inference → 8 normalized values
        # 3. Unscale to original image pixel coords
        # 4. Return DetectionResult(corners=(4,2), detector_name="corner_cnn")
```

Default model path: `models/releases/sudoku_corner_cnn.onnx` (version-less canonical name, same pattern as CNN OCR).

### Detector chain (`backend/api/routes.py`)

```
Photo path: CornerCnnDetector → OpenCV fallback
Screenshot path: OpenCV only (unchanged)
```

## File Changes

### New files

| File | Responsibility |
|---|---|
| `backend/cv/corner_cnn_detector.py` | ONNX inference, implements GridDetector protocol |
| `backend/cv/corner_cnn_arch.py` | PyTorch model definition (MobileNetV3-Small + regression head) |
| `scripts/train_corner_cnn.py` | Training script with MLflow, freeze/unfreeze, augmentation |
| `scripts/migrate_yolo_to_corner.py` | One-time migration: 16-kpt YOLO → 4-corner CSV |
| `data/corner_cnn/` | Training data directory |

### Modified files

| File | Change |
|---|---|
| `backend/api/routes.py` | Detector chain: CornerCnnDetector → OpenCV |
| `ml_backend/app.py` | `/predict` returns 4 corners; `/fit` writes CSV; loads CornerCnnDetector |
| `scripts/labeling_pipeline.sh` | Train: corner-cnn replaces YOLO. Release: delete yolo, add corner-cnn |
| `scripts/release_model.py` | Delete `release_yolo()`, add `release_corner_cnn()` |

### Deleted

| File/logic | Reason |
|---|---|
| `scripts/train_yolo_grid.py` | Replaced by `train_corner_cnn.py` |
| `release_model.py: release_yolo()` | No longer publish YOLO |

### Unchanged

| File | Reason |
|---|---|
| `backend/cv/detector.py` | Interface unchanged |
| `backend/cv/yolo_detector.py` | Class retained; camera_monitor may still reference it |
| `backend/ocr/*` | CNN OCR pipeline untouched |
| `screen_monitor/*` | Uses OpenCV path, unaffected |
