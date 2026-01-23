# Sudoku Solver

A web application that solves Sudoku puzzles from images. Upload a photo or screenshot of a Sudoku puzzle, and the app will detect the grid, recognize the digits, and solve the puzzle.

## Features

- **Image Upload**: Upload Sudoku images via drag-and-drop or file selection
- **Grid Detection**: Computer vision pipeline to detect and extract the Sudoku grid
- **OCR**: Digit recognition using Tesseract OCR
- **Solving**: Backtracking algorithm to solve the puzzle
- **Web Interface**: Simple, responsive frontend

## Tech Stack

- **Backend**: Python + FastAPI
- **Computer Vision**: OpenCV (cv2)
- **OCR**: Tesseract
- **Frontend**: Vanilla HTML/CSS/JavaScript

## Installation

### Prerequisites

1. Python 3.11+
2. [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system

#### Installing Tesseract

**macOS**:
```bash
brew install tesseract
```

**Ubuntu/Debian**:
```bash
sudo apt-get install tesseract-ocr
```

**Windows**:
Download installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### Setup with uv

```bash
# Install dependencies
uv sync

# Install dev dependencies
uv sync --all-extras
```

### Setup with pip

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
# Using uv
uv run uvicorn backend.main:app --reload

# Using python directly
python -m uvicorn backend.main:app --reload
```

The application will be available at http://localhost:8000

## API Endpoints

- `GET /` - Web interface
- `GET /health` - Health check
- `POST /api/v1/sudoku:solve` - Solve a Sudoku from JSON
- `POST /api/v1/sudoku:solveImage` - Solve a Sudoku from an image
- `POST /api/v1/sudoku:detectGrid` - Detect grid from an image
- `GET /docs` - API documentation

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
│   │   └── digit_reader.py    # OCR digit recognition
│   ├── solver/
│   │   └── backtracking.py    # Sudoku solving algorithm
│   ├── models/
│   │   └── schemas.py         # Pydantic models
│   └── main.py                # FastAPI app
├── frontend/
│   ├── index.html             # Web UI
│   ├── style.css              # Styles
│   └── app.js                 # Frontend logic
├── tests/
│   ├── test_solver.py
│   ├── test_cv.py
│   └── test_ocr.py
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
# Using uv
uv run pytest

# Using pytest directly
pytest
```

## Limitations

- Requires clear, well-lit images
- Grid must be roughly rectangular in the image
- Works best with standard Sudoku formatting
- Tesseract accuracy varies with image quality

## License

MIT
