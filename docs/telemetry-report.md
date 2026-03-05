# Screen Monitor Telemetry Report

Session: 2026-03-05, ~2.5 min (one complete sudoku game)

## Summary

| Metric | Value |
|--------|-------|
| Total frames | 530 |
| Session duration | ~150s |
| Average FPS | 3.2 |
| Givens detected (avg) | 57.2 |

## Frame State Distribution

| State | Frames | % | Description |
|-------|--------|---|-------------|
| idle | 348 | 65.7% | No puzzle detected on screen |
| active | 89 | 16.8% | Puzzle detected, grid changing |
| stable | 78 | 14.7% | Grid stable, solution rendered |
| lost | 15 | 2.8% | Puzzle lost after detection |

65% idle 说明大部分时间屏幕上没有数独棋盘（打开/切换页面、解题后关闭等）。
active + stable 合计 31.5%，是实际在处理数独的时间。

## Pipeline Step Latency

| Step | Executions | Avg (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Total (ms) |
|------|-----------|----------|----------|----------|----------|------------|
| capture | 501 | 85.6 | 83.6 | 222.0 | 244.4 | 42,873 |
| detect | 171 | 121.6 | 61.7 | 95.1 | 228.7 | 20,798 |
| ocr | 87 | 26.6 | 32.2 | 48.2 | 49.6 | 2,316 |
| solve | 87 | 117.0 | 2.6 | 4.9 | 7,825.0 | 10,182 |

### Key Findings

1. **Capture 是最大瓶颈** — p50=83.6ms, p95=222ms，单步就占了 56% 的总耗时。
   每次循环都要截屏，这是帧率上限的主要约束。
   优化方向：降低截屏分辨率、只截取感兴趣区域（ROI）、或使用更快的截屏 API。

2. **Detect 的 p50 vs avg 差异大** — avg=121.6ms 但 p50=61.7ms，说明有长尾。
   p99=228.7ms 存在少量慢检测。可能是图像变化较大时 contour detection 计算量增加。

3. **OCR 最快且最稳定** — p50=32.2ms, p95=48.2ms，方差很小。
   CNN v2.0 的 ONNX 推理表现良好，不是瓶颈。

4. **Solve 有极端长尾** — p50=2.6ms 但 p99=7,825ms（~8秒）。
   大部分情况下求解器瞬间完成（回溯法对简单棋盘很快），但首次识别到完整棋盘时可能触发了一次高耗时求解。
   avg=117ms 被少量长尾拉高（87次调用中 p95 仍只有 4.9ms）。

5. **没有 cache hit 数据** — detect cache 和 solve cache 均未命中。
   这可能是因为棋盘一直在变化（用户在填数），每帧都是新状态。

## Givens Distribution

| Metric | Value |
|--------|-------|
| Samples | 87 |
| Average | 57.2 |
| p50 | 57.3 |
| p95 | 91.6 |

p50=57 表示典型识别到约 57 个已填数字（标准数独 81 格）。
p95=91.6 超过 81，说明有时 OCR 将空格误识别为数字，存在 false positive。

## Optimization Priorities

| Priority | Target | Current | Potential Impact |
|----------|--------|---------|-----------------|
| 1 | **Screen capture** | p50=84ms | 降低分辨率或 ROI 裁剪可提升 FPS 到 ~6-8 |
| 2 | **Detect cache** | 0% hit | 添加图像哈希快速比较，跳过未变化的帧 |
| 3 | **Solve cold start** | p99=8s | 首次求解加超时或并行化 |
| 4 | **OCR accuracy** | givens p95>81 | 调高置信度阈值减少 false positive |

## Time Budget Per Frame (p50)

```
capture  ████████████████████████████████████████████  83.6ms  (52%)
detect   ████████████████████████████████              61.7ms  (38%)
ocr      ████████████████                              32.2ms  (20%)
solve    █                                              2.6ms   (2%)
─────────────────────────────────────────────────────
total                                                 180.1ms → ~5.6 FPS theoretical max
```

Actual FPS (~3.2) is lower than theoretical max due to:
- Frame-to-frame overhead (state management, Qt signals, overlay rendering)
- Idle frames still consume capture time
- Occasional long-tail latencies
