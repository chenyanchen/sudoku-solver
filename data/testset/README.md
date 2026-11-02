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

2026-10-14: Corner CNN `20261014.2` (heatmap model) solved the plastic-sleeve photo that OpenCV fails, so the web path now retries with Corner CNN's corners whenever OpenCV's grid does not solve uniquely. Score 14/15 → 15/15; the 60 training images go from 59/60 to 60/60 (`IMG_8712`), 0 wrong.

2026-11-02: added 41 uploads from the public site (2026-10-15 to 11-01), kept in `local/`: 33 phone photos of a phone app from one user (glare, moiré; 17 to 50 givens), 6 of the owner's test uploads, 2 from other visitors. 89 reviewed uploads were split by puzzle (a hash of the solution, or of the givens when ambiguous, mod 3 == 0 goes here), so photos of one puzzle stay on one side; the other 48 went to training. Every digit was checked against the image; corners are the detector's, confirmed by a line search in the warp or corrected by hand. Excluded: a killer sudoku, a grid covered by a dialog, a 16×16 puzzle. OCR 20261029.1 reads all 56 exactly through the web path. Corner CNN 20261016.1: median corner error 1.1% on the earlier 15, 1.7% on the 41 new (9 above 3%).
