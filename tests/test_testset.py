"""The independent test set must stay well-formed and attributable."""

import json
from pathlib import Path

from backend.solver.backtracking import SudokuSolver, is_valid_grid

ROOT = Path(__file__).resolve().parents[1] / "data" / "testset"
LABELS = json.loads((ROOT / "labels.json").read_text())
SOURCES = json.loads((ROOT / "sources.json").read_text())


def test_labels_are_valid_puzzles():
    for name, grid in LABELS.items():
        assert is_valid_grid(grid), name
        # 0 solutions means a transcription error; >1 is allowed (published ambiguous puzzle).
        assert SudokuSolver().count_solutions(grid) >= 1, name


def test_committed_images_exist_and_are_attributed():
    for name in LABELS:
        if name.startswith("local/"):
            continue  # gitignored, may be absent
        assert (ROOT / name).is_file(), name
        source = SOURCES[name]
        assert source["url"] and source["license"] and source["author"], name


def test_no_test_image_is_used_for_training():
    training = json.loads((ROOT.parent / "labels" / "sudoku_labels.json").read_text())
    assert not {Path(n).name for n in LABELS} & {Path(n).name for n in training}
    assert not [g for g in LABELS.values() if g in training.values()]
