# Sudoku Solver 机器学习复习指南

> 基于当前工作树代码、模型元数据和 Git 历史整理。复习时以“当前实现”为准；`README.md`、`CLAUDE.md` 和早期设计文档中有少量内容已经过时。

## 1. 一张图看懂系统

```text
输入图片/视频帧
  │
  ├─ 棋盘定位
  │    ├─ Web/照片：Corner CNN → OpenCV fallback
  │    ├─ 屏幕监控：OpenCV
  │    └─ Camera prototype：YOLO26n-pose
  │
  ├─ 透视变换：四角 → 450×450 正视棋盘
  ├─ 切分：9×9 → 81 个 cell
  ├─ OCR：48×48 灰度 cell → CNN → blank/数字 1~9
  ├─ 规则后处理：置信度阈值、二次识别、数独约束 rerank
  ├─ 有界修复：尝试低置信候选组合
  └─ 回溯求解器：验证并求解
```

这个仓库不是单一模型，而是一个**学习模型 + 传统 CV + 领域约束**的混合系统：

- 学习模型负责难以手写规则的视觉映射；
- OpenCV 负责几何变换、图像清洗和 fallback；
- 数独规则负责纠正部分 OCR 错误；
- ONNX Runtime 负责生产推理，PyTorch 只用于训练/导出。

---

## 2. 当前有哪些模型

| 模型 | 任务 | 输入 | 输出 | 当前主要位置 |
|---|---|---|---|---|
| OCR CNN | 10 分类 | `1×48×48` 灰度 cell | 10 个 logits：0=空白，1~9=数字 | `backend/ocr/model_arch.py` |
| Corner Regression CNN | 四角坐标回归 | `3×256×256` RGB letterbox | 8 个 `[0,1]` 坐标 | `backend/cv/corner_cnn_arch.py` |
| YOLO26n-pose（历史/相机原型） | 棋盘检测 + 关键点 | `3×640×640` RGB letterbox | bbox、置信度、4/16 个关键点 | `backend/cv/yolo_detector.py` |

当前 canonical release：

- OCR：`models/releases/sudoku_digit_cnn.onnx` → `sudoku_digit_cnn_20260402.1.onnx`
- Corner CNN：`models/releases/sudoku_corner_cnn.onnx` → `sudoku_corner_cnn_20260402.3.onnx`
- YOLO 推理代码仍保留，但当前仓库没有 `yolo_grid_detect.onnx` release。

---

## 3. OCR CNN：任务定义与数据

### 3.1 任务

每个数独 cell 被视为一个 10 分类问题：

```text
class 0 = blank
class 1..9 = digit 1..9
```

模型输出 logits，推理端自行做 softmax：

\[
p_i = \frac{e^{z_i-\max(z)}}{\sum_j e^{z_j-\max(z)}}
\]

减去最大 logit 是数值稳定技巧，避免指数溢出。

### 3.2 当前数据规模

`data/labels/sudoku_labels.json` 当前包含：

- 38 张已标注棋盘；
- 3,078 个 cell；
- 其中 blank 1,808 个，数字样本共 1,270 个；
- 各数字约 134~151 个，blank 明显占多数。

训练样本不是直接存储的 cell 图片，而是在训练时：

```text
原图 → 棋盘检测/warp → 450×450 → 切 81 格 → clean_cell(48×48)
```

因此，**检测器与预处理也是 OCR 数据分布的一部分**。换检测器、warp 方法或 cell 清洗方法，即使 OCR 网络不变，也可能造成 domain shift。

### 3.3 cell 预处理

`backend/cv/cell_extractor.py::clean_cell`：

1. 转灰度；
2. resize 到 48×48；
3. 3×3 Gaussian blur；
4. CLAHE 局部对比度增强；
5. Otsu 二值化；
6. morphology open 去噪；
7. dilation 加粗细笔画；
8. 转成白底黑字。

训练和推理复用同一个 `clean_cell`，这是避免 train-serving skew 的关键。

---

## 4. OCR CNN 模型结构

### 4.1 v1：较大的全连接头

历史结构：

```text
Input 1×48×48
  → Conv 1→32, 3×3 + BN + ReLU + MaxPool     # 24×24
  → Conv 32→64, 3×3 + BN + ReLU + MaxPool    # 12×12
  → Conv 64→128, 3×3 + BN + ReLU + MaxPool   # 6×6
  → Flatten: 128×6×6 = 4608
  → Linear 4608→256 + ReLU + Dropout(0.25)
  → Linear 256→10
```

问题：参数主要集中在 `4608→256` 全连接层，模型约 4.9 MB，空间结构被直接 flatten，尺寸也被固定死。

### 4.2 v2：`custom_small_cnn`

当前仍保留，可用于旧 checkpoint：

```text
Input 1×48×48
  → [Conv 1→32, BN, ReLU, Pool, Dropout(0.1)]
  → [Conv 32→64, BN, ReLU, Pool, Dropout(0.1)]
  → [Conv 64→128, BN, ReLU, Pool]
  → AdaptiveAvgPool2d(1)
  → Linear 128→64 + ReLU + Dropout(0.25)
  → Linear 64→10
```

参数量：**102,026**。

核心改动：用 Global/Adaptive Average Pooling 代替大 Flatten 层。

收益：

- 参数量和模型大小约下降 91%；
- 减少全连接层过拟合；
- 卷积特征被压成每通道一个统计量；
- ONNX release 从约 4.9 MB 降到约 422 KB，精度基本不变。

### 4.3 v3：`custom_cnn_v3`（当前 OCR 架构）

```text
Input 1×48×48
  → Stem: Conv 1→24, 5×5 + BN + ReLU + Pool     # 24×24
  → ResBlock 24→48 + SE
  → MaxPool                                      # 12×12
  → ResBlock 48→96 + SE
  → MaxPool                                      # 6×6
  → AdaptiveAvgPool2d(1)
  → Dropout(0.25)
  → Linear 96→10
```

参数量：**169,414**。

#### Residual Block

每个 block：

```text
主支路：Conv3×3 → BN → ReLU → Conv3×3 → BN → SE
捷径：Identity；通道变化时用 Conv1×1 投影
输出：ReLU(主支路 + 捷径)
```

残差连接学习的是相对输入的修正：

\[
y = F(x) + x
\]

通道数不同时，使用投影 `W_s x`：

\[
y = F(x) + W_sx
\]

作用：改善梯度传播，让更深的特征提取比简单堆叠卷积更容易训练。

#### Squeeze-and-Excitation（SE）

1. Squeeze：对每个通道做全局平均池化；
2. Excitation：两层 MLP 生成 0~1 通道权重；
3. Scale：每个 feature map 乘对应权重。

```text
C×H×W → GAP → C → Linear(C,C/4) → ReLU
      → Linear(C/4,C) → Sigmoid → 通道加权
```

直觉：模型学会“当前数字识别更应该关注哪些通道”。

#### 为什么 stem 用 5×5

相比 3×3，5×5 一开始就有更大的感受野，更容易捕获数字笔画的整体形状；代价是少量额外参数。

### 4.4 `timm_mobilenetv3` 备选

- `mobilenetv3_small_100`；
- 单通道输入；
- 随机初始化，`pretrained=False`；
- 参数量约 **1,527,818**。

它目前只是训练参数中的备选架构，不是 release 默认。对只有数千个简单灰度 cell 的任务，当前自定义 v3 更小，也更符合任务规模。

---

## 5. OCR 模型演进时间线

| 时间/版本 | 变化 | 动机/结果 |
|---|---|---|
| 2026-02，v1.1 | 首个 CNN OCR，替换传统 OCR 路线 | 建立 48×48、blank+1~9 分类、ONNX 推理链路 |
| v1.2 | 增加新棋盘数据 | 通过 hard-case 数据提升覆盖 |
| 2026-03，v2.0 | Flatten 大头改为 GAP 小头；加入 Dropout、label smoothing、cosine LR | 模型约缩小 91%，精度保持；102K 参数 |
| v2.1 | 加入 NYTimes 样本；提高原图 re-warp 质量 | 修复细网格线被误识别为数字 1、漏识别数字 4 |
| v3.x | ResBlock + SE + 5×5 stem | 169K 参数；历史评估中 false positive 从 5 降到 1 |
| v3.2 | 加入 NYTimes、装饰边框样本 | 历史提交记录 100% validation accuracy，但见第 13 节的数据切分风险 |
| v3.3 | 用 YOLO warp 生成训练输入 | 使训练分布贴近当时的 YOLO-first 推理管线 |
| v3.4 | MLflow 训练/发布体系下的当前训练版本 | 当前 release 为 `20260402.1`，训练版本元数据为 `3.4.0` |

版本制度也发生过变化：

- 早期：`vMAJOR.MINOR`，架构变化升 MAJOR，数据/超参数变化升 MINOR；
- 当前 release：统一使用 `YYYYMMDD.N`，训练版本另存为 `training_version`。

---

## 6. OCR 训练方法

实现：`scripts/train_cnn_ocr.py`。

### 6.1 数据增强

只对训练集随机执行：

- 亮度/对比度；
- Gaussian blur；
- erosion，模拟细笔画；
- dilation，模拟粗笔画；
- Gaussian noise。

没有旋转/翻转：数字方向具有语义，随意翻转会制造错误标签。

### 6.2 类别不平衡

blank 数量远高于单个数字类别。训练使用逆频率 class weight：

\[
w_c \propto \frac{1}{n_c}
\]

再归一化，使 10 个类别的权重和约等于 10。这样可避免模型只靠预测 blank 获得高 accuracy。

### 6.3 Loss、优化器和调度器

- Loss：weighted cross entropy；
- Label smoothing：0.1；
- Optimizer：AdamW；
- 默认 LR：`1e-3`；
- Weight decay：`1e-4`；
- Scheduler：CosineAnnealingLR；
- 默认 60 epochs；
- Early stopping patience：12。

Label smoothing 将 one-hot 目标变“软”，降低模型过度自信：

\[
y'_k=(1-\epsilon)y_k+\epsilon/K
\]

### 6.4 验证指标

- `accuracy`：所有 10 类整体准确率；
- `macro_f1_digits`：只对 1~9 分别算 F1 后平均；
- `blank_precision`：预测为空白的 cell 中，实际为空白的比例；
- 每个数字的 F1；
- 10×10 confusion matrix。

选择最佳模型的组合分数：

\[
score = 0.65F1_{digits} + 0.20Precision_{blank} + 0.15Accuracy
\]

数字 F1 权重最高，因为系统最怕给定数字识别错；一个错误 given 可能直接让整盘无解。

---

## 7. OCR 推理不是一次 argmax

实现：`backend/ocr/cnn_digit_reader.py`。

### 7.1 三阶段识别

#### Phase 0：OpenCV blank 先验

根据 24×24 图像的标准差和 ink ratio 判断“看起来是否为空”：

- `std < 4` → likely blank；
- Otsu 反色后 ink ratio `< 2.5%` → likely blank。

它只作为先验，不直接取代 CNN。

#### Phase 1：81 格批量 ONNX 推理

一次 batch 推理 81 个 cell，比循环调用 81 次 runtime 更省开销。

默认阈值：

- 数字：最高类为 1~9 且 confidence ≥ 0.55；
- 空白：最高类为 0、confidence ≥ blank threshold，且 OpenCV 也认为 likely blank。

#### Phase 2：不确定 cell 二次识别

对不确定 cell 做：

```text
Gaussian blur → adaptive threshold → 必要时反色 → erosion
```

然后再 batch inference。目的是恢复抗锯齿、很细的数字笔画。

### 7.2 Sudoku 约束 rerank

对置信度低于 0.80 的数字：

1. 若当前数字违反行/列/宫约束，先清零；
2. 按低置信度优先；
3. 从 top-k 数字候选中找概率 ≥ 0.15 且满足数独约束的值；
4. 最多迭代两轮；
5. 不会把最初判断为空白的格子强行填成数字。

这是典型的**模型概率 + 结构化领域知识**后处理。

### 7.3 Decode（`backend/ocr/pipeline.py`）

只有**解唯一**的识别结果才被信任。否则 `decode` 会：

- 多解（漏识别已知数）：在 CNN 认为可能有数字的空格里补数字；
- 无解（认错已知数）：优先选冲突格和低置信非零格，替换为 top candidates 或 blank；
- 最多修改 `max_changes`（默认 2）格，修改数从少到多枚举；
- 找到第一个解唯一的组合即停止。

它不是另一个 ML 模型，而是利用 CNN top-k 分布进行有界搜索。

---

## 8. Corner Regression CNN

### 8.1 为什么从 YOLO 转向直接回归

YOLO26n-pose 是通用的“检测 + 姿态关键点”模型，而本项目每张图通常只有一个已知结构的棋盘。历史设计判断：

- YOLO 约 2.7M 参数；
- 只有 38 张标注图时难以泛化；
- bbox、类别分数、NMS 等能力对单棋盘任务是额外复杂度；
- ImageNet 预训练 backbone + 直接回归四角更 data-efficient。

因此 Web 照片路径改为 Corner CNN，YOLO 训练脚本已删除。

### 8.2 当前结构

```text
Input: RGB 3×256×256
  → MobileNetV3-Small backbone, ImageNet pretrained
  → forward_features
  → AdaptiveAvgPool2d(1)
  → Linear 576→64
  → ReLU
  → Dropout(0.2)
  → Linear 64→8
  → Sigmoid
```

输出顺序：

```text
[x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl]
```

实测参数量：

- 总参数：**1,555,304**；
- backbone：1,517,856；
- regression head：37,448。

输出使用 sigmoid，把预测限制到 `[0,1]`。坐标相对于 256×256 letterbox 图像，而不是原图；推理后需去掉 padding 并除以 scale 映射回原图。

### 8.3 为什么保持 RGB

第一层改成灰度只能省很少参数，却无法直接完整利用 ImageNet RGB 预训练权重。38 张图片的数据规模下，预训练先验比少量参数节省更重要。

### 8.4 数据

`data/corner_cnn/labels.csv` 当前有 38 个样本，每行保存四个归一化角点。

初始数据从 YOLO 16-keypoint 标签迁移：

```text
P00 → TL, P03 → TR, P33 → BR, P30 → BL
```

Label Studio 后端可继续写入/更新 CSV，并把源图复制到 `data/corner_cnn/images/`。

### 8.5 数据增强

几何增强必须同步变换图片和角点：

- random perspective，最大约 8% 扰动；
- rotation ±15°；
- scale 0.8~1.2；
- translate ±10%；
- horizontal flip，并交换 `TL↔TR`、`BL↔BR`。

光度增强不改标签：

- brightness/contrast；
- Gaussian blur；
- random grayscale。

### 8.6 训练

- Loss：SmoothL1Loss；
- Optimizer：AdamW；
- Phase 1：前 5 epoch 冻结 backbone，只训练 head，LR `1e-3`；
- Phase 2：解冻 backbone，backbone LR `1e-5`，head LR `1e-4`；
- 默认 batch size 8；
- early stopping patience 15；
- 按最高 validation `warp_iou` 保存 checkpoint。

Smooth L1 在小误差时像 L2，梯度平滑；大误差时像 L1，对异常值不如纯 MSE 敏感。

### 8.7 指标

- `corner_mae`：8 个归一化坐标的平均绝对误差；
- `warp_iou`：预测四边形与 GT 四边形的 mask IoU。

`warp_iou` 更接近业务目标：角点有少量像素偏差不一定影响最终 450×450 warp，但四边形覆盖区域错了会直接破坏 cell extraction。

当前三个 release 的元数据：

| Release | warp_iou |
|---|---:|
| `20260402.1` | 0.5882 |
| `20260402.2` | 0.5880 |
| `20260402.3` | **0.5916** |

当前 canonical 指向 `.3`。这些结果离早期设计目标 `>0.95` 仍很远，符合“小数据是主要瓶颈”的判断。

### 8.8 推理置信度的含义

Corner CNN 没有输出分类置信度。`last_confidence` 是后处理构造的**几何合理性分数**，依据：

- 四边形是否凸；
- 原始预测是否越界；
- 是否过度贴近图片边缘。

它只适合 active learning 排序，不是校准后的概率。

---

## 9. YOLO26n-pose 的历史角色

### 9.1 4-keypoint 阶段

模型检测一个 sudoku bbox，并预测：

```text
TL, TR, BR, BL
```

相比 OpenCV，它更适合复杂背景、光照和倾斜照片。

### 9.2 16-keypoint 阶段

关键点扩展成 4×4 的宫格边界交点：

```text
 0 ─ 1 ─ 2 ─ 3
 │   │   │   │
 4 ─ 5 ─ 6 ─ 7
 │   │   │   │
 8 ─ 9 ─10 ─11
 │   │   │   │
12 ─13 ─14 ─15
```

外角索引为 `[0, 3, 15, 12]`。设计目标是对 9 个宫分别做 warp，以适应装饰棋盘的不均匀 cell。

### 9.3 训练方式（历史实现）

- Ultralytics YOLO26n-pose；
- 输入 640×640；
- 预训练权重迁移学习；
- 亮度、饱和度、轻旋转、平移、缩放、透视、mosaic 增强；
- 不做水平/垂直翻转，避免数字镜像；
- 由 Ultralytics 记录 box/pose loss、precision、recall、mAP。

### 9.4 当前状态

- Web：已经由 Corner CNN 取代；
- YOLO 训练和 release 流程已删除；
- `YoloGridDetector` 和 camera monitor 仍保留；
- camera monitor 仍要求 YOLO ONNX，但当前仓库没有对应模型文件。

因此复习时应把 YOLO 看作“重要演进阶段 + 尚存的相机原型依赖”，而不是当前 Web 主路径。

---

## 10. 传统 CV 与 ML 如何分工

### 10.1 棋盘检测

当前 Web detector chain：

```text
CornerCnnDetector → OpenCVGridDetector
```

学习模型失败时，传统轮廓检测兜底。`ChainDetector` 让下游只依赖统一的 `DetectionResult`。

### 10.2 透视变换

无论角点来自 CNN、YOLO 还是 OpenCV，统一交给 `warp_grid()` 变成 450×450。检测与 warp 解耦，使模型可以替换而不改 OCR。

### 10.3 OCR

```text
OpenCV clean_cell → CNN 分类 → 数独规则 rerank → solver repair
```

这是端到端深度学习之外的另一种实用路线：让模型只解决局部视觉问题，让确定性规则保证全局结构。

---

## 11. 实验追踪、导出与发布

### 11.1 MLflow

当前实验：

- `sudoku-cnn-ocr`；
- `sudoku-corner-cnn`。

每次 run 记录：

- 超参数；
- batch/epoch metrics；
- Git commit tag；
- 最佳 `.pt` checkpoint；
- OCR confusion matrix。

本地查看：

```bash
uv run --extra ml mlflow ui --backend-store-uri mlruns --port 5000   # from the repo root
```

### 11.2 PyTorch → ONNX

训练产物 `.pt` 包含：

- `state_dict`；
- 训练版本；
- input size；
- OCR 还包含 arch、class values、best score；
- Corner CNN 包含 best warp IoU。

导出：

- OCR：ONNX opset 17，动态 batch，可选 INT8 dynamic quantization；
- Corner CNN：ONNX opset 18，动态 batch；
- 部署统一使用 ONNX Runtime CPU provider。

### 11.3 Release

```bash
./scripts/labeling_pipeline.sh release cnn \
  --version 20260402.2 --run-id <run_id>

./scripts/labeling_pipeline.sh release corner-cnn \
  --version 20260402.4 --run-id <run_id>
```

发布脚本：

1. 从 MLflow 下载指定 run 的 `.pt`；
2. 转 ONNX；
3. 生成带日期的 immutable artifact；
4. 更新 canonical symlink；
5. 写入 run ID、metric、Git commit 和发布时间元数据。

这里没有自动发布“最新”或“最好”run，必须显式指定 `run-id`，避免错误实验被静默上线。

---

## 12. Active Learning / Label Studio 闭环

```text
现有 release 预标注
  → Label Studio 人工纠正
  → /fit 写回四角 CSV 或 OCR JSON
  → 重新训练并记录到 MLflow
  → 人工比较指标
  → 显式 release
  → 新模型继续预标注
```

两类标注：

- Corner project：4 个 keypoints（TL/TR/BR/BL）；
- OCR project：9 行文本形式的 9×9 grid。

OCR 预标注 score 只有在识别网格可被求解时才保留平均置信度，否则置 0；这会把“视觉置信度”和“领域有效性”结合起来用于人工排序。

---

## 13. 复习时必须能指出的风险与局限

### 13.1 OCR 验证集存在同图泄漏风险

当前 `stratified_split` 是按 **cell** 分层随机切分，不是按原始棋盘分组。同一张图的 81 个 cell 很可能同时进入训练集和验证集。

后果：背景、字体、渲染风格高度相似，历史上的 100% validation accuracy 可能明显乐观。更可靠的方法是按 image/group split，再检查每类覆盖。

### 13.2 数据量很小

- OCR：38 张棋盘；
- Corner CNN：38 张图片。

数据增强不能完全替代真实分布覆盖，尤其是：不同字体、装饰边框、反光、透视、遮挡、低分辨率和不同设备。

### 13.3 Corner CNN 指标仍低

当前最佳 release `warp_iou≈0.592`，离设计目标 0.95 较远。Web 仍需要 OpenCV fallback，不能把当前 Corner CNN 当成已解决问题。

### 13.4 设计文档与实现有漂移

以代码为准的典型差异：

- Corner 设计文档写了 cosine scheduler，当前训练代码没有 scheduler；
- 设计写“按图片类型 stratified split”，当前实现是普通随机 image split；
- `README.md` 仍有已删除的 `--output` 训练参数和旧模型名；
- `labeling_pipeline.sh train` 仍用 `--use-yolo` 训练 OCR，但当前默认 YOLO release 不存在；
- release sidecar 使用 `model_version`，而 OCR reader 读取的是 `version`，当前 reader 可能回退为文件名作为版本。

### 13.5 指标要对齐业务

单 cell accuracy 不等于整盘成功率。若每个 given 的准确率为 99%，一盘 30 个 givens 全部正确的概率粗略为：

\[
0.99^{30}\approx 0.74
\]

因此更应关注：

- whole-grid exact match；
- puzzle solvable rate；
- false-positive blank / false digit；
- 按场景分组的指标；
- detector + OCR 的端到端成功率。

### 13.6 “置信度”不是都能互相比较

- OCR confidence：softmax 最大值，仍未必校准；
- YOLO confidence：检测头分数；
- Corner confidence：人工构造的几何 sanity score；
- Label Studio OCR score：可解时才使用平均 OCR confidence。

名字相同，不代表统计含义相同。

---

## 14. 建议复习顺序

### 第一轮：建立全局图

1. 本文第 1、2、10 节；
2. `backend/api/routes.py`：看 detector → warp → OCR；
3. `backend/ocr/pipeline.py`：看 OCR → decode → solve。

目标：能口述端到端数据流，并区分 ML、CV、规则算法的职责。

### 第二轮：吃透 OCR

1. `backend/ocr/model_arch.py`；
2. `scripts/train_cnn_ocr.py`；
3. `backend/ocr/cnn_digit_reader.py`；
4. `backend/ocr/pipeline.py`。

目标：能画出 v2/v3 网络，解释 class weight、label smoothing、F1、batch inference、rerank。

### 第三轮：吃透检测模型演进

1. `backend/cv/corner_cnn_arch.py`；
2. `scripts/train_corner_cnn.py`；
3. `backend/cv/corner_cnn_detector.py`；
4. `backend/cv/yolo_detector.py`；
5. `docs/superpowers/specs/2026-04-01-corner-regression-cnn-design.md`。

目标：解释为什么 YOLO → direct regression，以及 letterbox 坐标如何映射回原图。

### 第四轮：工程化

1. `scripts/export_*_onnx.py`；
2. `scripts/release_model.py`；
3. `ml_backend/`；
4. `scripts/labeling_pipeline.sh`。

目标：讲清楚训练、实验追踪、标注、导出、发布和生产推理的闭环。

---

## 15. 自测题

1. 为什么 OCR 要把 blank 也作为模型类别，而不是只靠 OpenCV 判断？
2. v2 用 GAP 代替 Flatten 为什么能减少约 91% 模型体积？
3. Residual connection 和 SE 分别解决什么问题？
4. 为什么 OCR 关注 digit macro-F1 和 blank precision，而不只看 accuracy？
5. 为什么几何增强必须同步更新 Corner CNN 的角点标签？
6. 为什么 Corner CNN 使用 RGB 和 ImageNet 预训练，而 OCR 使用灰度小 CNN？
7. letterbox 后的归一化角点如何映射回原图？
8. Corner CNN 的 `last_confidence` 为什么不能解释成“预测正确概率”？
9. 为什么 99% cell accuracy 仍可能只有约 74% 的整盘全对率？
10. 当前 OCR validation 为什么可能数据泄漏？应该怎样 split？
11. YOLO 16-keypoint 相比 4-keypoint 原本想解决什么问题？
12. 为什么当前 Web 仍保留 OpenCV fallback？
13. ONNX Runtime 与 PyTorch 在这个仓库中分别承担什么角色？
14. Sudoku constraint rerank 和 grid repair 有什么区别？
15. 如果下一轮只能做一件事提升效果，为什么通常应优先扩大并正确划分真实数据，而不是继续加深网络？

---

## 16. 速记卡

```text
OCR current arch: custom_cnn_v3
Input/output: 1×48×48 → 10 logits (blank + 1..9)
Params: 169,414
Key blocks: 5×5 stem + 2×(ResBlock+SE) + GAP
Loss: weighted CE + label smoothing 0.1
Optimizer: AdamW; scheduler: cosine
Best score: .65 digit macro-F1 + .20 blank precision + .15 accuracy
Inference: clean → batch → uncertain recheck → Sudoku rerank → bounded repair

Corner current arch: ImageNet MobileNetV3-Small + 64-d regression head
Input/output: 3×256×256 → 8 normalized coords
Params: 1,555,304; head 37,448
Loss: SmoothL1
Training: freeze head warmup → low-LR full fine-tune
Primary metric: quadrilateral warp IoU
Current best release: 20260402.3, warp_iou≈0.5916

Historical detector: YOLO26n-pose, 4 kpt → 16 kpt
Current Web: Corner CNN → OpenCV fallback
Current Screen: OpenCV
Current Camera prototype: YOLO + OCR CNN
Runtime: ONNX Runtime CPU
Training: PyTorch/timm/Ultralytics + MLflow
Biggest ML concern: tiny dataset + OCR cell-level split leakage
```
