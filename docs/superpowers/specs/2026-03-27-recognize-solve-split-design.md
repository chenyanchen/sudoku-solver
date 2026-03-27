# Recognize + Solve API Split — Design Spec

将 `solveImage` 拆为 `recognize`（检测 + OCR）和现有 `solve`（求解），删除 `solveImage`。

## API 变更

### `POST /api/v1/sudoku:recognize` (新)

上传图片，返回识别到的数独棋盘。

**Request:** `multipart/form-data`, field `image`

**Response:**

```json
{
  "success": true,
  "message": "Recognized 28 givens",
  "grid": [[5,3,0,...], ...],
  "detected_image": "base64...",
  "detector_name": "yolo",
  "ocr_confidence": 0.92,
  "ocr_engine": "cnn",
  "ocr_model_version": "v3.3.1",
  "ocr_latency_ms": 45.2
}
```

字段说明：
- `grid` — 9×9 int 数组，0 表示空格（原 `original_grid`）
- `detected_image` — warp 后棋盘 base64 PNG
- `detector_name` — 哪个检测器找到的（yolo / opencv）
- `ocr_confidence` — OCR 平均置信度（原 `confidence`）
- `ocr_engine`, `ocr_model_version`, `ocr_latency_ms` — OCR 元数据

失败时 `success=false`，`grid` 为 null。

### `POST /api/v1/sudoku:solve` (已有，不改)

接受 JSON grid 求解。保持现有接口不变。

### `DELETE /api/v1/sudoku:solveImage` (删除)

移除此端点。

## Schema 变更

- 新增 `RecognizeResponse`（上述字段）
- 删除 `ImageSolveResponse`
- 现有 `SolveRequest` / `SolveResponse` 不变，但 `SolveRequest.grid.cells` 考虑重命名为 `grid`（与 recognize 一致）

## 字段重命名汇总

| 旧 | 新 | 原因 |
|----|-----|------|
| `original_grid` | `grid` | 领域术语，更简洁 |
| `confidence` | `ocr_confidence` | 明确是 OCR 置信度 |
| `solved_grid` | （移除） | recognize 不求解 |

## 文件变更

| Action | File |
|--------|------|
| Modify | `backend/models/schemas.py` — 新增 `RecognizeResponse`，删除 `ImageSolveResponse` |
| Modify | `backend/api/routes.py` — 新增 `recognize` 端点，删除 `solveImage` 端点，清理 `_image_response` helper |
| Modify | `frontend/app.js` — 调用 `recognize` + `solve` 两步 |
