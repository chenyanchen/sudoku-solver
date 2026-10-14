# Independent test set

Images that were **never used for training or tuning**, for measuring the
full pipeline (`detect → OCR → decode → solve`):

```bash
uv run python scripts/eval_cnn_ocr.py --labels data/testset/labels.json
```

- `images/`: redistributable images (CC / GPL). Attribution is in `sources.json`.
- `local/`: images that may not be redistributed, such as website screenshots. This folder is gitignored and its labels are kept in `labels.json`. If the images are missing, eval skips those labels.
- `corners.csv`: grid corners (TL, TR, BR, BL, normalized), proposed by the OpenCV detector and reviewed on the warps.
- `labels.json`: givens as printed (0 = blank), transcribed by hand. A label must have at least one solution. A published puzzle with several solutions is kept on purpose: the expected outcome is "no unique solution".

Rules: never add these images to `data/labels/sudoku_labels.json` (enforced by `tests/test_testset.py`). Report the score before and after any change that was made because of a test-set failure.

History: the first 13 images scored 8/13. Two fixes came from those failures: hidden-single solving (a 17-clue puzzle exceeded the search budget) and the OpenCV-first detector order (grids cropped flush to the image edge). After them the score is 12/13.

2026-10-14: added 16 browser screenshots, kept in `local/` (sudoku.com, websudoku, puzzle-sudoku, NYT, livesudoku and 4 Poki games; easy to evil; tight, wide and low-res crops). All 16 were read exactly with no decode corrections, so the total is 28/29. Clean web renders are an easy domain. Still missing: dark themes, phone app screenshots, and more photos.

2026-10-14: added 4 phone photos of screens (`local/IMG_9748–9751.jpg`; AARP, Baidu Baike, Sudoku Kingdom, a Chinese sudoku site; circled highlights). They scored 3/4: `IMG_9749` failed detection, because at full resolution moiré merged the grid outline with a neighbouring frame and the detector returned a single 3×3 box. Fix: OpenCV detection also tries a ≤1280px-wide copy and prefers a regular grid at any scale. After it: 4/4, total 32/33; the 40 training-set images are unchanged at 35/40.

2026-10-14: split for Corner CNN training. 18 of the 33 images moved to the training set (`data/raw/images/`, `local/` there when not redistributable): the 4 phone photos `IMG_9748–9751`; every other CC image in name order (gnome-debian-lenny, oceans17, 8381070829, 17-clue-r929, first-discovered, flickr-sudoku-paper-puzzle); and 8 screenshots, a share of each site, with both `nyt-easy-1` crops kept together. 15 remain: 7 CC images, including 3 photos, and 8 screenshots. Before the split the remaining 15 scored 14/15; the plastic-sleeve photo still fails.
