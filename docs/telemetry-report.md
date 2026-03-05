# Screen Monitor Telemetry Report

## Session Overview

Data source: Prometheus (Docker, localhost:9091), scraping `sudoku-screen-monitor` on `:9092`.
Three sessions detected on 2026-03-05:

| Session | Time (UTC) | Duration | Frames | Avg FPS | Max FPS | Notes |
|---------|-----------|----------|--------|---------|---------|-------|
| 1 | 09:01–09:03 | ~2.2 min | 414 | 3.0 | 5.0 | Before optimization |
| 2 | 09:44–09:51 | ~7.5 min | 1644 | 3.5 | 5.8 | Partial optimization (no cache propagation) |
| 3 | 10:36–10:40 | ~3.5 min | 629 | 2.9 | 4.2 | Full optimization |

Session 2 ran between commits — telemetry was enhanced but cache-hit propagation
and solver timeout guard were not yet active. Session 3 includes all optimizations.

---

## Before vs After (Session 1 → Session 3)

### Pipeline Step Latency

| Step | Metric | Before (S1) | After (S3) | Change |
|------|--------|-------------|------------|--------|
| **capture** | p50 | 87.0 ms | 88.1 ms | +1.3% (unchanged, expected) |
| | p95 | 209.5 ms | 223.5 ms | — |
| | p99 | 241.9 ms | 244.7 ms | — |
| **detect** | p50 | 51.0 ms | 40.9 ms | **-19.8%** |
| | p95 | 84.7 ms | 233.5 ms | +176% (long tail, see notes) |
| | p99 | 197.1 ms | 990.3 ms | (outliers from content-region upscale) |
| **ocr** | p50 | 32.4 ms | 31.6 ms | -2.5% (stable) |
| | p95 | 48.2 ms | 48.2 ms | — |
| **solve** | p50 | 2.6 ms | 15.5 ms | +496% (see notes) |
| | p95 | 5.0 ms | 76.3 ms | — |
| | p99 | 15.7 ms | 1903.7 ms | (timeout guard capping, was 7825 ms in S2) |

### Key Changes

| Metric | Before (S1) | After (S3) | Impact |
|--------|-------------|------------|--------|
| FPS avg | 3.0 | 2.9 | Roughly flat |
| FPS max | 5.0 | 4.2 | — |
| Cache detect hits | 0 | 5 | Cache now observable |
| Cache solve hits | 0 | **54** | Significant reuse |
| Givens avg | 55.1 | 54.2 | Stable |
| Givens p95 | **87.1** | **73.1** | **-16.1%** — fewer false positives |
| Givens p99 | ~91 | ~75 | Well below 81 ceiling |

### Frame State Distribution

| State | Before (S1) | After (S3) |
|-------|-------------|------------|
| idle | 69.1% | 64.4% |
| active | 15.9% | 11.1% |
| stable | 15.0% | **22.9%** |
| lost | 0.0% | 1.6% |

Stable frames rose from 15% to 23% — the pipeline holds solutions longer.

---

## Optimization Effectiveness

### Unit 1: Frame Downscale (detect p50 -20%)

Downscaling Retina frames (2560→1280) before detection reduced detect p50 from
51 ms to 41 ms. The warped grid output is always 450x450 (perspective transform
normalizes), so OCR quality is unaffected.

Detect p95/p99 increased because Session 3 may have had more complex scenes
triggering Stage 2 (content-region upscale) and Stage 3 (CLAHE retry) fallbacks.
The p50 improvement is the meaningful signal.

### Unit 2: Solver Timeout (solve p99: 7825 → 1904 ms)

The 1M iteration cap worked — Session 2 (without the guard) still had a p99 of
7925 ms; Session 3 capped at 1904 ms. The timeout is triggering on pathological
grids from OCR misreads. The solver still returns None and the pipeline falls
through to repair or skips the frame.

Solve p50 rose from 2.6 ms to 15.5 ms. This is likely due to the iteration
counter overhead plus different puzzle difficulty in Session 3 (more stable
frames = more solve attempts on harder states).

### Unit 3: CLAHE Reuse

Eliminated 81 `createCLAHE()` calls per grid. Not directly measurable in step
latency (folded into OCR), but OCR avg stayed flat at ~27 ms despite other
changes — confirming no regression.

### Unit 4: OCR Blank Detection (givens p95: 87 → 73)

Tightening blank thresholds (`blank_threshold` 0.65→0.75, `ink_ratio` 1.5%→2.5%)
reduced false positives. Givens p95 dropped from 87 to 73 — now well below the
81-cell ceiling. This means fewer unsolvable grids from noise, fewer wasted
solve + repair cycles.

### Unit 5: Cache Hit Propagation (detect=5, solve=54)

Cache hits are now observable. Session 3 recorded:
- 5 detect cache hits (frame content unchanged between scans)
- 54 solve cache hits (same grid signature seen across stable frames)

The solve cache is working well — 54 hits out of 159 solve-eligible frames (34%
hit rate). This avoids redundant OCR + solve when the grid hasn't changed.

### Unit 6: Pre-allocated OpenCV Objects

Structuring elements (`_MORPH_KERNEL_5x5`, `_MORPH_KERNEL_15x15`) pre-allocated.
Minor per-frame savings, not individually measurable.

---

## Time Budget Per Frame (p50, Session 3)

```
capture  █████████████████████████████████████████████  88.1ms  (50%)
detect   █████████████████████                          40.9ms  (23%)
ocr      ████████████████                               31.6ms  (18%)
solve    ████████                                       15.5ms   (9%)
─────────────────────────────────────────────────────
total                                                  176.1ms → ~5.7 FPS theoretical max
```

Capture remains the dominant cost at 50%. Further FPS gains require:
- Faster screen capture API (CGWindowListCreateImage vs mss)
- ROI-only capture (grab only the grid region, not full screen)
- Reducing capture frequency when grid is stable (adaptive polling)

---

## Remaining Issues

1. **Detect p95/p99 regression** — Long tail increased, likely from fallback
   detection stages on complex scenes. Consider adding a timeout or early-exit
   to Stage 2/3 when Stage 1 finds near-square candidates.

2. **Solve p50 increase** — 2.6→15.5 ms. Investigate whether iteration counter
   overhead is significant, or if this is purely puzzle-difficulty variance.
   Could also be caused by more diverse grid states reaching the solver due to
   better OCR (fewer false positives → more valid-looking grids to solve).

3. **Capture latency unchanged** — mss screen grab is the floor. This is an
   OS-level constraint. Consider platform-specific capture APIs for macOS.

4. **FPS plateau** — Despite pipeline speedups, FPS stayed ~3.0. The bottleneck
   is capture + idle frame processing, not the pipeline itself. With 88 ms
   capture + overhead, ~6 FPS is the hard ceiling. Adaptive frame skipping or
   async capture could help.
