# Independent test set

Images that were **never used for training or tuning**, for measuring the
full pipeline (`detect → OCR → decode → solve`):

```bash
uv run python scripts/eval_cnn_ocr.py --labels data/testset/labels.json
```

- `images/`: redistributable images (CC / GPL). Attribution is in `sources.json`.
- `local/`: images that may not be redistributed, such as website screenshots. This folder is gitignored and its labels are kept in `labels.json`. If the images are missing, eval skips those labels.
- `labels.json`: givens as printed (0 = blank), transcribed by hand. A label must have at least one solution. A published puzzle with several solutions is kept on purpose: the expected outcome is "no unique solution".

Rules: never add these images to `data/labels/sudoku_labels.json` (enforced by `tests/test_testset.py`). Report the score before and after any change that was made because of a test-set failure.

History: the first 13 images scored 8/13. Two fixes came from those failures: hidden-single solving (a 17-clue puzzle exceeded the search budget) and the OpenCV-first detector order (grids cropped flush to the image edge). After them the score is 12/13.

2026-10-14: added 16 browser screenshots, kept in `local/` (sudoku.com, websudoku, puzzle-sudoku, NYT, livesudoku and 4 Poki games; easy to evil; tight, wide and low-res crops). All 16 were read exactly with no decode corrections, so the total is 28/29. Clean web renders are an easy domain. Still missing: dark themes, phone app screenshots, and more photos.
