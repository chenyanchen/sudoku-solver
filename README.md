# Sudoku Solver

A web application that solves Sudoku puzzles from images. Upload a photo or screenshot of a Sudoku puzzle, and the app will detect the grid, recognize the digits, and solve the puzzle.

## Features

- **Image Upload**: Upload Sudoku images via drag-and-drop or file selection
- **Grid Detection**: Computer vision pipeline to detect and extract the Sudoku grid
- **OCR**: Self-trained CNN digit recognition (ONNX Runtime, CPU first)
- **Solving**: Backtracking algorithm to solve the puzzle
- **Web Interface**: Simple, responsive frontend

## Tech Stack

- **Backend**: Python + FastAPI
- **Computer Vision**: OpenCV (cv2)
- **OCR Runtime**: ONNX Runtime
- **OCR Training**: PyTorch (+ optional timm/Optuna/MLflow)
- **Frontend**: Vanilla HTML/CSS/JavaScript

## Installation

### Prerequisites

1. Python 3.11+

### Setup with uv

```bash
# Install dependencies
uv sync

# Install dev dependencies
uv sync --extra dev

# Install ML stack for CNN train/export/inference
uv sync --extra ml
```

## CNN OCR Workflow

### 1. Prepare labels

Default labels file:

`data/labels/sudoku_labels.json`

Format:

```json
{
  "sudoku_2.png": [[...9x9...]],
  "sudoku_3.png": [[...9x9...]]
}
```

Training screenshots are stored under `data/raw/images/`.

### 2. Train model

```bash
uv run python scripts/train_cnn_ocr.py \
  --labels data/labels/sudoku_labels.json \
  --output models/checkpoints/sudoku_digit_cnn.pt \
  --version 1.0.0
```

### 3. Export ONNX

```bash
uv run python scripts/export_cnn_onnx.py \
  --checkpoint models/checkpoints/sudoku_digit_cnn.pt \
  --output models/releases/sudoku_digit_cnn_v1.0.onnx
```

### 4. Evaluate model

```bash
uv run python scripts/eval_cnn_ocr.py \
  --labels data/labels/sudoku_labels.json \
  --cnn-model models/releases/sudoku_digit_cnn_v1.0.onnx
```

## Running the Application

```bash
uv run uvicorn backend.main:app --reload
```

The application will be available at http://localhost:8000

### Runtime configuration

```bash
# OCR engine: only cnn is supported
export OCR_ENGINE=cnn

# CNN model and thresholds
export CNN_MODEL_PATH=models/releases/sudoku_digit_cnn_v1.1.onnx
export CNN_BLANK_THRESHOLD=0.65
export CNN_DIGIT_THRESHOLD=0.55
export CNN_RERANK_CONFIDENCE=0.80
export CNN_TOPK_CANDIDATES=4
export CNN_REPAIR_MAX_CHANGES=2
export CNN_REPAIR_MAX_CELLS=14
```

## API Endpoints

- `GET /` - Web interface
- `GET /health` - Health check
- `POST /api/v1/sudoku:solve` - Solve a Sudoku from JSON
- `POST /api/v1/sudoku:solveImage` - Solve a Sudoku from an image
- `POST /api/v1/sudoku:detectGrid` - Detect grid from an image
- `GET /docs` - API documentation

`/api/v1/sudoku:solveImage` response now includes optional OCR metadata:

- `ocr_engine`
- `ocr_model_version`
- `ocr_latency_ms`

## Project Structure

```
sudoku-solver/
├── backend/
│   ├── api/
│   │   └── routes.py          # API endpoints
│   ├── cv/
│   │   ├── preprocessor.py    # Image preprocessing
│   │   ├── grid_detector.py   # Grid detection
│   │   └── cell_extractor.py  # Cell extraction
│   ├── ocr/
│   │   ├── cnn_digit_reader.py # CNN OCR inference
│   │   └── grid_repair.py      # OCR candidate-based auto-repair
│   ├── solver/
│   │   └── backtracking.py    # Sudoku solving algorithm
│   ├── models/
│   │   └── schemas.py         # Pydantic models
│   └── main.py                # FastAPI app
├── data/
│   ├── labels/
│   │   ├── sudoku_labels.json         # Training/evaluation labels
│   │   └── sudoku_label_sources.json  # Label provenance
│   └── raw/
│       └── images/                    # Raw Sudoku screenshots
├── models/
│   ├── checkpoints/           # Training checkpoints (.pt)
│   └── releases/              # Deployable ONNX artifacts
├── scripts/
│   ├── train_cnn_ocr.py       # CNN training
│   ├── export_cnn_onnx.py     # ONNX export
│   └── eval_cnn_ocr.py        # Offline evaluation
├── frontend/
│   ├── index.html             # Web UI
│   ├── style.css              # Styles
│   └── app.js                 # Frontend logic
├── tests/
│   ├── test_solver.py
│   ├── test_cv.py
│   ├── test_cnn_ocr.py
│   └── test_grid_repair.py
├── pyproject.toml
└── README.md
```

## Usage

1. Open the web interface at http://localhost:8000
2. Upload an image of a Sudoku puzzle
3. Click "Solve Puzzle"
4. View the detected grid and solution

## Testing

```bash
uv run pytest
```

## Limitations

- Requires clear, well-lit images
- Grid must be roughly rectangular in the image
- Works best with standard Sudoku formatting
- CNN quality depends on labeled data coverage and hard-case mining

## License

MIT
