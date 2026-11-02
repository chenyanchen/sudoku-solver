"""Render synthetic 16x16 Sudoku grids (symbols 1-9 and A-G) for OCR training.

Real 16x16 uploads are rare, so the 16x16 OCR model learns its letters here.
Output has the layout of the real data, so the OCR training script reads it
with the same warp → binarize → cut path as real photos:

    <out>/images/synth16-NNNNN.jpg
    <out>/labels.json   "images/synth16-NNNNN.jpg" -> 16x16 grid (0 blank, 10-16 = A-G)
    <out>/corners.csv   grid corners, normalized TL, TR, BR, BL

Usage:
    uv run --extra ml python scripts/synth_16x16_grids.py --count 1500 --fonts data/fonts --out data/synth_16x16
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIZE, BOX = 16, 4
SYMBOLS = "123456789ABCDEFG"  # value v is SYMBOLS[v - 1]
# Sans, serif, slab, monospace, condensed, rounded and hand-drawn faces: with
# 22 sans faces only, held-out serif and monospace grids rarely read. Some
# families are kept out on purpose to measure how well other fonts read:
# Arial Rounded, Georgia, Comic Sans, Menlo, Monaco, SF Mono, Optima, Skia,
# Courier New and Impact.
_SYS, _SUP = "/System/Library/Fonts/", "/System/Library/Fonts/Supplemental/"
SYSTEM_FONTS = [_SYS + f for f in (
    "Helvetica.ttc", "HelveticaNeue.ttc", "SFNS.ttf", "SFNSRounded.ttf", "SFCompact.ttf",
    "SFCompactRounded.ttf", "Avenir.ttc", "Avenir Next.ttc", "Avenir Next Condensed.ttc",
    "Geneva.ttf", "LucidaGrande.ttc", "NewYork.ttf", "Palatino.ttc", "Courier.ttc",
    "MarkerFelt.ttc", "Noteworthy.ttc", "Hiragino Sans GB.ttc", "ヒラギノ角ゴシック W3.ttc",
    "ヒラギノ角ゴシック W6.ttc", "STHeiti Medium.ttc",
)] + [_SUP + f for f in (
    "Arial.ttf", "Arial Bold.ttf", "Arial Black.ttf", "Arial Narrow.ttf", "Tahoma.ttf",
    "Tahoma Bold.ttf", "Verdana.ttf", "Verdana Bold.ttf", "Trebuchet MS.ttf", "Trebuchet MS Bold.ttf",
    "DIN Alternate Bold.ttf", "DIN Condensed Bold.ttf", "Futura.ttc", "GillSans.ttc",
    "Microsoft Sans Serif.ttf", "PTSans.ttc", "Seravek.ttc", "Times New Roman.ttf",
    "Times New Roman Bold.ttf", "PTSerif.ttc", "Baskerville.ttc", "Charter.ttc", "Cochin.ttc",
    "Didot.ttc", "Hoefler Text.ttc", "Iowan Old Style.ttc", "STIXTwoText.ttf", "Rockwell.ttc",
    "SuperClarendon.ttc", "AmericanTypewriter.ttc", "Andale Mono.ttf", "PTMono.ttc", "Chalkboard.ttc",
)]


def solved_grid(rng: random.Random) -> list[list[int]]:
    """A random valid 16x16 solution: a pattern, shuffled within its symmetries."""
    def shuffled_axis() -> list[int]:
        bands = rng.sample(range(BOX), BOX)
        return [b * BOX + r for b in bands for r in rng.sample(range(BOX), BOX)]

    rows, cols, symbols = shuffled_axis(), shuffled_axis(), rng.sample(range(1, SIZE + 1), SIZE)
    return [[symbols[(BOX * (r % BOX) + r // BOX + c) % SIZE] for c in cols] for r in rows]


def puzzle(rng: random.Random) -> list[list[int]]:
    """Givens of a solution: real 16x16 puzzles show about 40-65% of the cells."""
    keep = rng.uniform(0.35, 0.7)
    return [[v if rng.random() < keep else 0 for v in row] for row in solved_grid(rng)]


def load_fonts(extra_dirs: list[Path]) -> list[str]:
    fonts = [f for f in SYSTEM_FONTS if Path(f).exists()]
    for directory in extra_dirs:
        fonts += [str(p) for p in sorted(directory.rglob("*.ttf")) if "Italic" not in p.name]
    if not fonts:
        raise SystemExit("No fonts found; pass --fonts <dir with .ttf files>")
    return fonts


def render(grid: list[list[int]], fonts: list[str], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Draw the grid flat, then photograph it: returns (BGR image, corners in pixels)."""
    cell = rng.randint(40, 90)
    side = cell * SIZE
    pad = int(side * rng.uniform(0.04, 0.12))
    dark = rng.random() < 0.1
    paper = tuple(rng.randint(20, 50) for _ in range(3)) if dark else tuple(rng.randint(225, 255) for _ in range(3))
    ink = tuple(rng.randint(200, 245) for _ in range(3)) if dark else tuple(rng.randint(0, 70) for _ in range(3))
    thin = tuple(rng.randint(70, 110) for _ in range(3)) if dark else tuple(rng.randint(150, 215) for _ in range(3))
    thick = tuple(rng.randint(150, 220) for _ in range(3)) if dark else tuple(rng.randint(40, 130) for _ in range(3))

    canvas = Image.new("RGB", (side + 2 * pad, side + 2 * pad), paper)
    draw = ImageDraw.Draw(canvas)
    # App-style highlights: a selected cell, its row, column and box.
    if rng.random() < 0.6:
        sr, sc = rng.randrange(SIZE), rng.randrange(SIZE)
        tint = tuple(max(0, min(255, p + rng.randint(-35, 10))) for p in paper)
        for r in range(SIZE):
            for c in range(SIZE):
                if r == sr or c == sc or (r // BOX == sr // BOX and c // BOX == sc // BOX):
                    draw.rectangle([pad + c * cell, pad + r * cell, pad + (c + 1) * cell, pad + (r + 1) * cell], fill=tint)
        if rng.random() < 0.5:
            box_col = tuple(rng.randint(60, 140) for _ in range(2)) + (rng.randint(180, 255),)
            x, y = pad + (sc // BOX) * BOX * cell, pad + (sr // BOX) * BOX * cell
            draw.rectangle([x, y, x + BOX * cell, y + BOX * cell], outline=box_col, width=rng.randint(2, 4))

    thin_w, thick_w = rng.randint(1, 2), rng.randint(2, 5)
    for k in range(SIZE + 1):
        width = thick_w if k % BOX == 0 else thin_w
        color = thick if k % BOX == 0 else thin
        p = pad + k * cell
        draw.line([(p, pad), (p, pad + side)], fill=color, width=width)
        draw.line([(pad, p), (pad + side, p)], fill=color, width=width)

    # Size by the ink of a capital-height glyph and centre each glyph by its
    # own ink box: font metrics (ascent, descent) differ too much between fonts.
    font_path = rng.choice(fonts)
    height = rng.uniform(0.45, 0.75) * cell
    probe = ImageFont.truetype(font_path, 100)
    left, top, right, bottom = draw.textbbox((0, 0), "8", font=probe)
    font = ImageFont.truetype(font_path, max(8, round(100 * height / (bottom - top))))
    # Numbers the player entered are often blue.
    entered = (rng.randint(150, 200), rng.randint(170, 220), 255) if dark else (rng.randint(20, 90), rng.randint(20, 90), rng.randint(150, 230))
    for r in range(SIZE):
        for c in range(SIZE):
            if grid[r][c]:
                color = entered if rng.random() < 0.08 else ink
                symbol = SYMBOLS[grid[r][c] - 1]
                left, top, right, bottom = draw.textbbox((0, 0), symbol, font=font)
                cx = pad + (c + 0.5 + rng.uniform(-0.06, 0.06)) * cell
                cy = pad + (r + 0.5 + rng.uniform(-0.06, 0.06)) * cell
                draw.text((cx - (left + right) / 2, cy - (top + bottom) / 2), symbol, font=font, fill=color)

    image = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    corners = np.float32([[pad, pad], [pad + side, pad], [pad + side, pad + side], [pad, pad + side]])
    return photograph(image, corners, rng)


def photograph(image: np.ndarray, corners: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Perspective, lighting, glare, blur, noise and JPEG, as in phone photos of screens."""
    h, w = image.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.float32([[rng.uniform(-0.06, 0.06) * w, rng.uniform(-0.06, 0.06) * h] for _ in range(4)])
    dst -= dst.min(axis=0)
    out_w, out_h = int(dst[:, 0].max()), int(dst[:, 1].max())
    matrix = cv2.getPerspectiveTransform(src, dst)
    border = tuple(rng.randint(0, 255) for _ in range(3))
    image = cv2.warpPerspective(image, matrix, (out_w, out_h), borderValue=border)
    corners = cv2.perspectiveTransform(corners[None], matrix)[0]

    out = image.astype(np.float32)
    yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    gradient = 1 + rng.uniform(-0.2, 0.2) * (xx / out_w - 0.5) + rng.uniform(-0.2, 0.2) * (yy / out_h - 0.5)
    out *= gradient[..., None]
    for _ in range(rng.choice([0, 0, 1, 2])):  # glare
        gx, gy, radius = rng.uniform(0, out_w), rng.uniform(0, out_h), rng.uniform(0.03, 0.15) * out_w
        out += (np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * radius ** 2)) * rng.uniform(40, 160))[..., None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    sigma = rng.uniform(0, 1.6)
    if sigma > 0.3:
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    noise = rng.uniform(0, 8)
    out = np.clip(out + np.random.default_rng(rng.randrange(2**32)).normal(0, noise, out.shape), 0, 255).astype(np.uint8)
    scale = rng.uniform(0.6, 1.4)
    out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, encoded = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(55, 95)])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR), corners * scale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fonts", type=Path, action="append", default=[], help="Extra .ttf directory (e.g. Roboto)")
    parser.add_argument("--out", type=Path, default=Path("data/synth_16x16"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    fonts = load_fonts(args.fonts)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    labels, rows = {}, ["image,x_tl,y_tl,x_tr,y_tr,x_br,y_br,x_bl,y_bl"]
    for i in range(args.count):
        grid = puzzle(rng)
        image, corners = render(grid, fonts, rng)
        name = f"synth16-{i:05d}.jpg"
        cv2.imwrite(str(args.out / "images" / name), image)
        h, w = image.shape[:2]
        labels[f"images/{name}"] = grid
        rows.append(name + "," + ",".join(f"{v:.6f}" for x, y in corners for v in (x / w, y / h)))
    (args.out / "labels.json").write_text(json.dumps(labels))
    (args.out / "corners.csv").write_text("\n".join(rows) + "\n")
    print(f"{args.count} grids, {len(fonts)} fonts -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
