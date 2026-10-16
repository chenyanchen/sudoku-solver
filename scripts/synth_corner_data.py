"""Render synthetic Sudoku images with exact grid corners for Corner CNN training.

Each image is a grid (random line weights, colors, fonts, shading, highlights)
on a page with the distractors real images have: keypads, neighbouring grids,
decorative frames, titles, text and UI blocks. Half are "screenshots" (axis
aligned, any crop); half are "photos" (page on a background, perspective,
rotation, paper bend, uneven light, glare, moiré, blur, noise). Corners are the
centres of the grid's outer lines, mapped through every geometric step.

Usage:
    uv run --extra ml python scripts/synth_corner_data.py --count 6000 --seed 0
Writes <out>/images/*.jpg and <out>/labels.csv (the sudoku_corners.csv schema).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
FONT_DIRS = [Path("/System/Library/Fonts"), Path("/System/Library/Fonts/Supplemental"), Path("/Library/Fonts"),
             Path("/usr/share/fonts")]
FONT_NAMES = [
    "Arial", "Arial Bold", "Arial Narrow", "Helvetica", "HelveticaNeue", "Georgia", "Georgia Bold", "Times New Roman",
    "Times New Roman Bold", "Courier New", "Courier New Bold", "Verdana", "Verdana Bold", "Trebuchet MS", "Tahoma",
    "Futura", "Avenir", "Avenir Next", "DIN Alternate Bold", "DIN Condensed Bold", "Comic Sans MS", "Impact",
    "Gill Sans", "GillSans", "Rockwell", "Baskerville", "Didot", "Menlo", "Monaco", "Chalkboard", "Marker Felt",
    "DejaVuSans", "DejaVuSerif", "LiberationSans-Regular", "LiberationSerif-Regular",
]


def find_fonts() -> list[Path]:
    found = []
    for name in FONT_NAMES:
        for d in FONT_DIRS:
            for ext in (".ttf", ".ttc", ".otf"):
                if (d / f"{name}{ext}").exists():
                    found.append(d / f"{name}{ext}")
                    break
            else:
                continue
            break
    return found


class Renderer:
    def __init__(self, rng: np.random.Generator, fonts: list[Path]):
        self.rng = rng
        self.fonts = fonts
        self._font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

    # ----- helpers -------------------------------------------------------
    def u(self, a: float, b: float) -> float:
        return float(self.rng.uniform(a, b))

    def p(self, prob: float) -> bool:
        return bool(self.rng.random() < prob)

    def font(self, size: int, path: Path | None = None) -> ImageFont.FreeTypeFont:
        path = path or (self.fonts[self.rng.integers(len(self.fonts))] if self.fonts else None)
        size = max(8, size)  # some fonts (Helvetica.ttc) fail to render below this
        key = (str(path), size)
        if key not in self._font_cache:
            self._font_cache[key] = (
                ImageFont.truetype(str(path), size) if path else ImageFont.load_default(size=size)
            )
        return self._font_cache[key]

    def color(self, base: tuple[int, int, int], jitter: int = 20) -> tuple[int, int, int]:
        return tuple(int(np.clip(c + self.rng.integers(-jitter, jitter + 1), 0, 255)) for c in base)

    # ----- grid ----------------------------------------------------------
    def style(self) -> dict:
        dark = self.p(0.12)
        paper = self.color((30, 32, 38), 15) if dark else self.color(
            [(255, 255, 255), (250, 248, 240), (238, 236, 228), (245, 240, 225), (230, 230, 230)][self.rng.integers(5)], 8)
        inks = [(20, 20, 20), (60, 60, 60), (110, 110, 110), (40, 50, 140), (60, 90, 200), (190, 60, 60),
                (230, 120, 120), (120, 40, 120), (110, 70, 40), (40, 110, 60)]
        ink = self.color((200, 200, 205), 30) if dark else self.color(inks[self.rng.integers(len(inks))], 15)
        return {"paper": paper, "ink": ink, "dark": dark}

    def draw_grid(self, page: np.ndarray, x0: float, y0: float, cell: float, st: dict, distractor: bool = False) -> None:
        """Draw a 9x9 grid whose outer line centres are at x0, y0 .. x0 + 9 cell."""
        r, s = self.rng, 4  # supersampling shift for anti-aliased lines
        thin = max(1.0, cell * self.u(0.015, 0.05))
        thick = thin * self.u(1.0, 3.5) if self.p(0.85) else thin
        outer = thick * self.u(1.0, 1.8)
        faint = self.p(0.3)
        thin_ink = tuple(int(c * 0.45 + p * 0.55) for c, p in zip(st["ink"], st["paper"])) if faint else st["ink"]
        img = Image.fromarray(page)
        dr = ImageDraw.Draw(img)

        def box(c0, r0, c1, r1, fill):
            dr.rectangle([x0 + c0 * cell, y0 + r0 * cell, x0 + c1 * cell, y0 + r1 * cell], fill=fill)

        tint = self.color(st["ink"], 0)
        mix = lambda a: tuple(int(t * a + p * (1 - a)) for t, p in zip(tint, st["paper"]))
        if self.p(0.25):  # alternating box shading
            for b in range(9):
                if b % 2 == 0:
                    box(b % 3 * 3, b // 3 * 3, b % 3 * 3 + 3, b // 3 * 3 + 3, mix(self.u(0.06, 0.18)))
        if self.p(0.35):  # selection highlights: row, column, box, cell
            rr, cc = r.integers(9), r.integers(9)
            light = mix(self.u(0.08, 0.2))
            box(0, rr, 9, rr + 1, light)
            box(cc, 0, cc + 1, 9, light)
            box(cc // 3 * 3, rr // 3 * 3, cc // 3 * 3 + 3, rr // 3 * 3 + 3, light)
            box(cc, rr, cc + 1, rr + 1, self.color((250, 200, 90), 30) if self.p(0.5) else mix(self.u(0.25, 0.45)))
        page[:] = np.asarray(img)

        # Lines, drawn with cv2 for sub-pixel anti-aliasing.
        def line(p, q, w, colour):
            cv2.line(page, (int(p[0] * 2 ** s), int(p[1] * 2 ** s)), (int(q[0] * 2 ** s), int(q[1] * 2 ** s)),
                     colour[::-1] if False else colour, max(1, int(round(w))), cv2.LINE_AA, s)

        for k in range(10):
            w = outer if k in (0, 9) else thick if k % 3 == 0 else thin
            colour = st["ink"] if k % 3 == 0 else thin_ink
            ext = outer / 2 if k in (0, 9) else 0
            line((x0 + k * cell, y0 - ext), (x0 + k * cell, y0 + 9 * cell + ext), w, colour)
            line((x0 - ext, y0 + k * cell), (x0 + 9 * cell + ext, y0 + k * cell), w, colour)

        # Digits.
        img = Image.fromarray(page)
        dr = ImageDraw.Draw(img)
        fpath = self.fonts[r.integers(len(self.fonts))] if self.fonts else None
        size = max(6, int(cell * self.u(0.45, 0.8)))
        f = self.font(size, fpath)
        given_ink = st["ink"] if self.p(0.6) else self.color((20, 20, 20) if not st["dark"] else (230, 230, 230), 10)
        user_ink = self.color((40, 90, 210), 25)
        density = self.u(0.22, 0.5)
        for rr in range(9):
            for cc in range(9):
                cx, cy = x0 + (cc + 0.5) * cell, y0 + (rr + 0.5) * cell
                if r.random() < density:
                    d = str(r.integers(1, 10))
                    colour = user_ink if r.random() < 0.15 else given_ink
                    dr.text((cx, cy), d, font=f, fill=colour, anchor="mm")
                    if r.random() < 0.04:
                        rad = cell * 0.35
                        dr.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=self.color((60, 80, 200), 40),
                                   width=max(1, int(cell * 0.04)))
                elif not distractor and r.random() < 0.06:  # pencil marks
                    sf = self.font(max(5, int(cell * 0.22)), fpath)
                    for m in r.choice(9, r.integers(1, 5), replace=False):
                        dr.text((cx + (m % 3 - 1) * cell * 0.3, cy + (m // 3 - 1) * cell * 0.3), str(m + 1), font=sf,
                                fill=thin_ink, anchor="mm")
        page[:] = np.asarray(img)

    # ----- page ----------------------------------------------------------
    def text_block(self, dr: ImageDraw.ImageDraw, x: float, y: float, w: float, h: float, colour, scale: float) -> None:
        line_h = max(6.0, scale * self.u(0.25, 0.6))
        yy = y
        f = self.font(max(5, int(line_h * 0.8)))
        while yy + line_h < y + h:
            if self.p(0.5) and self.fonts:
                words = "".join(self.rng.choice(list("abcdefghijklmnopqrstuvwxyz      ")) for _ in range(int(w / line_h * 1.6)))
                dr.text((x, yy), words, font=f, fill=colour)
            else:
                dr.rectangle([x, yy + line_h * 0.2, x + w * self.u(0.4, 1.0), yy + line_h * 0.7], fill=colour)
            yy += line_h * self.u(1.2, 1.8)

    def page(self) -> tuple[np.ndarray, np.ndarray]:
        """A page with one target grid; returns (RGB page, 4x2 corners)."""
        r = self.rng
        st = self.style()
        cell = self.u(18, 70)
        g = 9 * cell
        tight = self.p(0.1)
        ml, mr, mt, mb = ((self.u(0, 0.02) * g for _ in range(4)) if tight
                          else (self.u(0.03, 1.2) * g for _ in range(4)))
        # "box": a 3x3 keypad beside the grid, the size of a few cells; Corner CNN
        # placed corners on such keypads, which look like the grid's 3x3 boxes.
        keypad = None if tight or not self.p(0.5) else r.choice(["right", "below", "row", "box", "rows"])
        if keypad == "right":
            mr = max(mr, g * self.u(0.25, 0.45))
        elif keypad == "box":
            mr = max(mr, g * self.u(0.45, 0.8))
        elif keypad in ("below", "row", "rows"):
            mb = max(mb, g * self.u(0.2, 0.45))
        W, H = int(ml + g + mr), int(mt + g + mb)
        page = np.empty((H, W, 3), np.uint8)
        page[:] = st["paper"]
        x0, y0 = ml, mt
        img = Image.fromarray(page)
        dr = ImageDraw.Draw(img)
        faint = tuple(int(c * 0.5 + p * 0.5) for c, p in zip(st["ink"], st["paper"]))
        if not tight:
            if self.p(0.4) and mt > g * 0.15:  # title / header
                if self.p(0.4):
                    dr.rectangle([0, 0, W, mt * self.u(0.3, 0.9)], fill=self.color(st["ink"], 30))
                f = self.font(max(8, int(min(mt * 0.5, g * 0.15))))
                dr.text((W * self.u(0.1, 0.5), mt * 0.5), self.rng.choice(["SUDOKU", "Sudoku", "Easy", "Hard", "数独", "Puzzle 12"]),
                        font=f, fill=st["ink"], anchor="mm")
            for _ in range(r.integers(0, 4)):  # text and UI blocks in the margins
                side = r.integers(4)
                if side == 0 and mt > 20:
                    self.text_block(dr, self.u(0, W / 2), self.u(0, mt * 0.5), self.u(0.2, 0.8) * W, mt * 0.4, faint, cell)
                elif side == 1 and mb > 20:
                    self.text_block(dr, self.u(0, W / 2), y0 + g + mb * 0.2, self.u(0.2, 0.8) * W, mb * 0.6, faint, cell)
                elif side == 2 and ml > 20:
                    self.text_block(dr, ml * 0.05, self.u(0, H / 2), ml * 0.8, self.u(0.2, 0.6) * H, faint, cell)
                elif side == 3 and mr > 20 and keypad not in ("right", "box"):
                    self.text_block(dr, x0 + g + mr * 0.1, self.u(0, H / 2), mr * 0.8, self.u(0.2, 0.6) * H, faint, cell)
            if keypad:
                kf = self.font(max(6, int(cell * 0.6)))
                fill = self.color(st["paper"], 25) if self.p(0.5) else None
                if keypad == "right":
                    kc = min(cell * self.u(0.8, 1.3), (mr - 10) / 1.2)
                    kx = x0 + g + (mr - kc) * self.u(0.2, 0.6)
                    for i in range(9):
                        ky = y0 + i * g / 9 + (g / 9 - kc) / 2
                        dr.rounded_rectangle([kx, ky, kx + kc, ky + kc * 0.95], radius=kc * 0.15, outline=st["ink"], fill=fill,
                                             width=max(1, int(cell * 0.03)))
                        dr.text((kx + kc / 2, ky + kc / 2), str(i + 1), font=kf, fill=st["ink"], anchor="mm")
                elif keypad == "box":
                    kc = min(g * self.u(0.12, 0.2), (mr - 10) / 3.4)
                    gap = kc * self.u(0.05, 0.25) if self.p(0.6) else 0.0  # gapless: a lined 3x3 grid
                    side_len = 3 * kc + 2 * gap
                    panels = 2 if self.p(0.35) and 2.4 * side_len < g else 1
                    kx = x0 + g + (mr - side_len) * self.u(0.1, 0.5)
                    ky = y0 + self.u(0, max(0.0, g - panels * side_len * 1.2))
                    width = max(1, int(cell * self.u(0.03, 0.08)))
                    for panel in range(panels):
                        py = ky + panel * side_len * 1.2
                        for i in range(9):
                            bx, by = kx + (i % 3) * (kc + gap), py + (i // 3) * (kc + gap)
                            if gap:
                                dr.rounded_rectangle([bx, by, bx + kc, by + kc], radius=kc * self.u(0, 0.2),
                                                     outline=st["ink"], fill=fill, width=width)
                            else:
                                dr.rectangle([bx, by, bx + kc, by + kc], outline=st["ink"], fill=fill, width=width)
                            dr.text((bx + kc / 2, by + kc / 2), str(i + 1), font=self.font(max(6, int(kc * 0.55))),
                                    fill=st["ink"], anchor="mm")
                else:
                    cols = {"row": 9, "rows": 6}.get(keypad, 3)
                    kc = min(g / cols * 0.9, (mb - 10) / {9: 1, 6: 2.3}.get(cols, 3.3))
                    for i in range(9 if cols != 6 else 12):
                        kx = x0 + (i % cols) * g / cols + (g / cols - kc) / 2
                        ky = y0 + g + 8 + (i // cols) * kc * 1.1
                        dr.rounded_rectangle([kx, ky, kx + kc, ky + kc], radius=kc * 0.15, outline=st["ink"], fill=fill,
                                             width=max(1, int(cell * 0.03)))
                        dr.text((kx + kc / 2, ky + kc / 2), str(i % 9 + 1), font=kf, fill=st["ink"], anchor="mm")
        page[:] = np.asarray(img)
        if not tight and self.p(0.2):  # neighbouring grid, cut by the page edge
            side = self.rng.choice(["l", "r"])
            off = self.u(0.15, 0.6) * cell
            nx = x0 - g - off if side == "l" else x0 + g + off
            if (side == "l" and ml > off + 5) or (side == "r" and mr > off + 5 and keypad not in ("right", "box")):
                self.draw_grid(page, nx, y0 + self.u(-0.3, 0.3) * g, cell, st, distractor=True)
        if not tight and self.p(0.2):  # decorative frame(s)
            img = Image.fromarray(page)
            dr = ImageDraw.Draw(img)
            m = cell * self.u(0.15, min(1.2, min(ml, mr, mt, mb) / cell * 0.9 + 0.15))
            fw = max(1, int(cell * self.u(0.04, 0.35)))
            colour = self.color(st["ink"], 60) if self.p(0.5) else self.color((60, 140, 60), 40)
            dr.rounded_rectangle([x0 - m, y0 - m, x0 + g + m, y0 + g + m], radius=self.u(0, m), outline=colour, width=fw)
            page[:] = np.asarray(img)
        self.draw_grid(page, x0, y0, cell, st)
        if self.p(0.12):  # rounded outer corners mask out the corner tips
            rad = cell * self.u(0.1, 0.4)
            for cx, cy in ((x0, y0), (x0 + g, y0), (x0 + g, y0 + g), (x0, y0 + g)):
                sx, sy = (1 if cx == x0 else -1), (1 if cy == y0 else -1)
                pts = np.array([[cx - sx * cell * 0.1, cy - sy * cell * 0.1], [cx + sx * rad, cy - sy * cell * 0.1],
                                [cx - sx * cell * 0.1, cy + sy * rad]], np.int32)
                cv2.fillConvexPoly(page, pts, st["paper"])
                cv2.ellipse(page, (int(cx + sx * rad), int(cy + sy * rad)), (int(rad), int(rad)), 0,
                            180 if sx > 0 and sy > 0 else 270 if sx < 0 and sy > 0 else 0 if sx < 0 else 90,
                            (180 if sx > 0 and sy > 0 else 270 if sx < 0 and sy > 0 else 0 if sx < 0 else 90) + 90,
                            st["ink"], max(1, int(cell * 0.04)), cv2.LINE_AA)
        corners = np.float32([[x0, y0], [x0 + g, y0], [x0 + g, y0 + g], [x0, y0 + g]])
        return page, corners

    # ----- scene ---------------------------------------------------------
    def background(self, h: int, w: int) -> np.ndarray:
        base = np.array(self.color([(120, 100, 80), (200, 200, 200), (60, 60, 70), (150, 120, 90), (230, 225, 215),
                                    (90, 110, 130)][self.rng.integers(6)], 40), np.float32)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        grad = 1 + self.u(-0.3, 0.3) * (xx / w - 0.5) + self.u(-0.3, 0.3) * (yy / h - 0.5)
        noise = cv2.resize(self.rng.normal(0, self.u(3, 25), (max(2, h // 16), max(2, w // 16))).astype(np.float32), (w, h))
        bg = base[None, None, :] * grad[..., None] + noise[..., None]
        for _ in range(self.rng.integers(0, 4)):  # objects on the table
            c = self.color((128, 128, 128), 120)
            x, y = int(self.u(0, w)), int(self.u(0, h))
            cv2.rectangle(bg, (x, y), (x + int(self.u(20, w / 2)), y + int(self.u(20, h / 2))), c, -1)
        return np.clip(bg, 0, 255).astype(np.uint8)

    def photo(self, page: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = self.rng
        h, w = page.shape[:2]
        if self.p(0.35):  # paper bend: y += a sin(pi x / w)
            a = self.u(-0.03, 0.03) * h
            xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            shift = a * np.sin(np.pi * xx / w)
            page = cv2.remap(page, xx, yy - shift, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            corners = corners + np.stack([np.zeros(4), a * np.sin(np.pi * corners[:, 0] / w)], 1).astype(np.float32)
        out_w = int(self.u(1.1, 2.0) * max(w, h))
        out_h = int(out_w * self.u(0.6, 1.6))
        scale = self.u(0.45, 0.95) * min(out_w / w, out_h / h)
        tilt = self.u(0, 0.35)
        ang = self.u(-20, 20)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        ctr = np.float32([out_w / 2 + self.u(-0.15, 0.15) * out_w, out_h / 2 + self.u(-0.15, 0.15) * out_h])
        dst = (src - [w / 2, h / 2]) * scale
        # Perspective: shrink one edge, as when the camera looks at the page at an angle.
        edge = r.integers(4)
        k = 1 - tilt
        if edge == 0:
            dst[[0, 1], 0] *= k
        elif edge == 1:
            dst[[1, 2], 1] *= k
        elif edge == 2:
            dst[[2, 3], 0] *= k
        else:
            dst[[3, 0], 1] *= k
        dst += self.rng.normal(0, 0.02 * scale * max(w, h), dst.shape)
        rot = cv2.getRotationMatrix2D((0, 0), ang, 1.0)[:, :2]
        dst = (dst @ rot.T + ctr).astype(np.float32)
        m = cv2.getPerspectiveTransform(src, dst)
        bg = self.background(out_h, out_w)
        warped = cv2.warpPerspective(page, m, (out_w, out_h))
        mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), m, (out_w, out_h))
        out = np.where(mask[..., None] > 127, warped, bg)
        corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), m).reshape(4, 2)
        return out, corners

    def photometric(self, img: np.ndarray, photo: bool) -> np.ndarray:
        h, w = img.shape[:2]
        f = img.astype(np.float32)
        if photo:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            light = 1 + self.u(-0.35, 0.2) * ((xx - self.u(0, w)) ** 2 + (yy - self.u(0, h)) ** 2) / (w * w + h * h)
            light += self.u(-0.25, 0.25) * (xx / w - 0.5)
            f *= light[..., None]
            f *= np.array([self.u(0.85, 1.1), self.u(0.9, 1.05), self.u(0.85, 1.15)], np.float32)
            if self.p(0.25):  # glare
                cx, cy, rx, ry = self.u(0, w), self.u(0, h), self.u(0.05, 0.3) * w, self.u(0.03, 0.2) * h
                g = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
                f += g[..., None] * self.u(60, 200)
            if self.p(0.2):  # shadow
                pts = (self.rng.random((4, 2)) * [w, h]).astype(np.int32)
                shadow = np.zeros((h, w), np.float32)
                cv2.fillPoly(shadow, [cv2.convexHull(pts)], 1.0)
                shadow = cv2.GaussianBlur(shadow, (0, 0), self.u(5, 40))
                f *= (1 - shadow * self.u(0.15, 0.45))[..., None]
            if self.p(0.12):  # moiré from photographing a screen
                period, theta = self.u(3, 9), self.u(0, math.pi)
                wave = np.sin((xx * math.cos(theta) + yy * math.sin(theta)) * 2 * math.pi / period)
                f *= (1 + self.u(0.04, 0.15) * wave)[..., None]
        if self.p(0.4 if photo else 0.1):
            k = self.u(0.4, 2.0)
            f = cv2.GaussianBlur(f, (0, 0), k)
        if photo or self.p(0.2):
            f += self.rng.normal(0, self.u(1, 10 if photo else 3), f.shape).astype(np.float32)
        return np.clip(f, 0, 255).astype(np.uint8)

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        page, corners = self.page()
        photo = self.p(0.5)
        if photo:
            img, corners = self.photo(page, corners)
        else:
            img = page
            if self.p(0.4):  # screenshot of a wider app: more page around the grid
                h, w = img.shape[:2]
                pad = [int(self.u(0, 0.6) * max(h, w)) for _ in range(4)]
                st = tuple(int(c) for c in img[0, 0])
                img = cv2.copyMakeBorder(img, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_CONSTANT, value=st)
                corners = corners + [pad[2], pad[0]]
        img = self.photometric(img, photo)
        # Final resolution: small web images up to downscaled camera frames.
        h, w = img.shape[:2]
        target = self.u(380, 1600)
        s = target / max(h, w)
        img = cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        corners = corners * [img.shape[1] / w, img.shape[0] / h]
        return img, corners.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render synthetic Sudoku images with exact corners")
    parser.add_argument("--count", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "synth_corner")
    args = parser.parse_args()

    fonts = find_fonts()
    print(f"{len(fonts)} fonts")
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.count):
        rng = np.random.default_rng([args.seed, i])
        renderer = Renderer(rng, fonts)
        while True:
            img, corners = renderer.sample()
            h, w = img.shape[:2]
            if (corners >= -0.5).all() and (corners[:, 0] <= w - 0.5).all() and (corners[:, 1] <= h - 0.5).all():
                break
        name = f"synth_{i:05d}.jpg"
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(args.out / "images" / name), bgr, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(55, 96))])
        norm = np.clip(corners / [w, h], 0, 1)
        rows.append([name] + [f"{v:.6f}" for v in norm.flatten()])
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{args.count}")
    with (args.out / "labels.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["image", "x_tl", "y_tl", "x_tr", "y_tr", "x_br", "y_br", "x_bl", "y_bl"])
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
