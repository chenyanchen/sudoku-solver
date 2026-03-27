# Unified YOLO Detection Pipeline — Design Spec

统一 YOLO 检测管线，支持截图、照片、装饰风格数独棋盘。

## 背景与动机

当前系统有两条检测路径：OpenCV 轮廓检测（截图）和 YOLO 4-keypoint（照片）。三种场景的支持现状：

| 场景 | 当前支持 | 瓶颈 |
|------|---------|------|
| 手机/电脑截图 | ✅ 良好 | — |
| 数独棋盘照片（拍照） | ⚠️ 部分 | YOLO 训练数据有限；CNN 仅 18 张图训练 |
| 卡通/装饰风格（竹制边框等） | ⚠️ 有限 | 固定 50px cell 切分无法适应不均匀间距；装饰网格线可能破坏轮廓检测 |

**目标**：统一为单一 YOLO 路径，支持全部三种场景，质量优先于速度。

## 方案选择

评估了三种方案：

- **方案 A（选定）**：单一 YOLO26n-pose 模型，分阶段替代 OpenCV
- 方案 B：双 YOLO 模型（nano 实时 + small 静态）— 维护成本翻倍，收益有限
- 方案 C：单 YOLO + 自适应前处理 — 输入类型检测不可靠，对 YOLO 增益有限

选择方案 A 的理由：YOLO26n 已验证实时可行；质量瓶颈在训练数据而非模型大小；单模型符合 Single Source of Truth。

## 分阶段交付

### Phase 1：YOLO 16-Keypoint + 自适应 Cell Extraction

**YOLO 模型升级**

16 个 keypoints 对应 3×3 宫格边界线的 4×4 交叉点：

```
 0 ── 1 ── 2 ── 3
 │    │    │    │
 4 ── 5 ── 6 ── 7
 │    │    │    │
 8 ── 9 ──10 ──11
 │    │    │    │
12 ──13 ──14 ──15
```

- 点 0/3/12/15 = 棋盘四角（与当前 4-kpt 兼容）
- 点 5/6/9/10 = 内部宫格交叉点
- 架构：YOLO26n-pose，`kpt_shape: [16, 2]`
- 输入：640×640 letterbox
- 迁移学习：从当前 4-kpt 权重初始化。当前模型 keypoint 顺序 [TL, TR, BR, BL]（索引 0-3）对应新模型索引 [0, 3, 15, 12]，权重映射按此对齐

**标注策略**

- 现有 4-kpt 标注 → 透视变换插值预生成剩余 12 点初始值
- Label Studio + `ml_backend` 预标注 → 人工微调
- 基础数据：`data/raw/images/` 现有 38 张图片，以此为起点标注 16-kpt
- 后续随使用场景扩展逐步积累更多数据

**自适应 Cell Extraction**

两级定位取代固定 50px 切分：

```
原图 → YOLO 16-kpt → 9 个宫格四边形 → 各自 warp 150×150 → 3×3 等分 → 81 个 cell
```

为什么两级而非直接 warp 450×450：直接 warp 假设 16 点网格均匀，但装饰棋盘宫格大小不一致。两级 warp 让每个宫格独立校正。

新接口：

```python
def extract_cells_adaptive(image: np.ndarray, keypoints_16: np.ndarray) -> list[np.ndarray]:
    """从原图 + 16 keypoints 提取 81 个 cell。

    Args:
        image: 原图（全分辨率）
        keypoints_16: shape (16, 2)，YOLO 输出的坐标（原图坐标系）
    Returns:
        81 个 cell 图像，行优先顺序
    """
```

旧 `extract_cells(warped)` 在 Phase 1-2 过渡期保留，Phase 3 移除。

**验证标准**

- 16 点平均偏差 < 5px（450×450 坐标系下）
- `data/raw/images/photo_1.jpg` 和 `data/raw/images/IMG_8710.jpg` 端到端正确识别

### Phase 2：CNN OCR 增强

**数据增强：双轨并行**

| 来源 | 内容 | 用途 |
|------|------|------|
| 公开数据集 | Chars74K、SVHN、合成印刷体等 | 预训练/增广，提升字体泛化 |
| 真实采集 | Label Studio 管线积累的截图 + 照片 cell | 微调，覆盖真实场景分布 |

公开数据需对齐到 CNN 输入格式：48×48 灰度、白底黑字、居中 + padding。

**训练流程**

```
1. 公开数据集 → 格式对齐脚本 → data/external_digits/
2. 真实数据 → Label Studio → data/labels/ + data/warped_grids/
3. 混合训练（二选一，由实验结果决定）：
   a) 公开数据预训练 → 真实数据微调
   b) 两者加权混合直接训练（真实数据权重更高）
```

**评估改进**

- `eval_cnn_ocr.py` 增加按场景类型分组（截图 / 照片 / 装饰风格）
- 新增 confusion matrix 输出，定位高频误识别对
- 验证集从训练集中严格分离

**模型版本**

架构不变（`custom_cnn_v3`），版本 MINOR bump（v3.5+）。MAJOR bump 仅限架构变更。

### Phase 3：统一管线，OpenCV 退役

**目标架构**

```
所有入口（web app / screen_monitor / camera_monitor）
    → YOLO 16-kpt
    → adaptive cell extraction
    → CNN OCR
    → solver
```

**screen_monitor 迁移**

| 当前 OpenCV 路径特性 | YOLO 路径等价方案 |
|---|---|
| ROI 记忆（只扫描 grid 附近区域） | YOLO 输入裁切到 ROI 区域 |
| visual hash 缓存（跳过未变化帧） | 保持不变，hash 在 warp 之前计算 |
| 跳过 CLAHE（截图光照均匀） | 同样跳过，CLAHE 仅在照片 cell 预处理中使用 |
| 帧稳定性追踪（N 帧一致才显示） | 保持不变，grid_tracker 逻辑与检测方式无关 |

**退役清理范围**

- 删除 `backend/cv/grid_detector.py`
- 删除 `backend/cv/preprocessor.py`
- 删除旧 `extract_cells(warped)` 接口
- 更新 `screen_monitor/solver_pipeline.py` 切换到 YOLO 路径
- 更新 `ml_backend/app.py` 使用 16-kpt 模型（Phase 1 即可同步适配）

**退役门控条件**

1. YOLO 在全部现有测试图片上检测成功率 ≥ OpenCV 当前水平
2. screen_monitor 实测帧率不低于当前 OpenCV 路径的 80%
3. `IMG_8710.jpg`（竹制棋盘）和 `photo_1.jpg` 端到端正确识别
