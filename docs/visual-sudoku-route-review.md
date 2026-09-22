# 视觉数独路线评估

> 后续诊断修正：部分照片数字标签与原图不符。本文的 `5/222` 等 OCR 数字是对旧错误标签的比较值，不是可信准确率。最新根因、标签复核及回归结果见 [普通照片求解进展](photo-solver-progress.md)。

## 结论与范围

保留“定位 → 几何校正 → 单格 OCR → 确定性求解”的混合路线。统一应发生在棋盘表示、OCR 和求解层，不要求截图、照片使用同一个定位器。当前优先级是恢复可信 OCR 基线、建立分层评估，再比较定位路线；不是立即更换 backbone。

本报告基于当前未提交工作树、release ONNX 实测及本地 MLflow 原始记录。未修改业务代码、权重或用户已有修改。以下 8 图是现有 Corner 验证集，混合截图与照片，不是独立照片测试集，也不保证对 OCR 训练独立。结果用于故障隔离，不用于宣称泛化性能。

## 已验证事实

### 1. 0.59 是 IoU，不是 accuracy

- `models/releases/sudoku_corner_cnn_20260402.3.meta.json`：`source_metric_name=warp_iou`，值 `0.5915717498871269`。
- `scripts/train_corner_cnn.py::compute_metrics`：256×256 上预测四边形与标注四边形的栅格 mask IoU；不是 rectified cell 对齐度。
- 发布 run 的 `params`：38 张图，30 train / 8 val，seed=42，val_ratio=0.2，epochs=120。
- 使用当前 ONNX 和同一切分重新推理：train IoU=0.6253326313，val IoU=0.5915717499；对应 MAE=0.0522121340 / 0.0529620424。训练集也不高，不能仅凭小数据认定为过拟合。

原始记录：`mlruns/316961318942125732/f31eeaf2ac0c44d9b7c4aab6bae45e56/`。

### 2. 当前 OCR 路径是更优先的故障

固定当前 `sudoku_digit_cnn.onnx`（解析到 `sudoku_digit_cnn_20260402.1.onnx`），使用标注四角 → `warp_grid` → `prepare_grid_for_ocr` → `extract_cells` → `CnnDigitReader.recognize_grid`。三个定位输入之外，其余处理保持不变。

| 验证图 | 标注给定数 | GT 四角下识别正确数 | Corner 四角下正确数 | OpenCV 四角下正确数 |
|---|---:|---:|---:|---:|
| IMG_8700.jpg | 26 | 2 | 0 | 1 |
| IMG_8705.jpg | 32 | 0 | 0 | 0 |
| IMG_8706.jpg | 23 | 0 | 1 | 1 |
| IMG_8707.jpg | 22 | 2 | 1 | 1 |
| sudoku_22.jpg | 35 | 0 | 1 | 0 |
| sudoku_5.jpg | 24 | 1 | 0 | 1 |
| sudoku_6.jpg | 22 | 0 | 1 | 0 |
| sudoku_7.jpg | 38 | 0 | 0 | 0 |
| 合计 | 222 | 5 | 4 | 4 |

这说明即使替换为标注角点，下游仍严重失败；不能通过提升检测器独自解决。

进一步隔离：`sudoku_5.jpg` 原图前三行与 JSON 标注目视一致；GT warp 后，跳过 recheck/rerank 的原始 argmax 只预测 3、7、8，给定数字命中 3/24。去掉 Web 额外 CLAHE 后仍为 3/24。不能把根因简单归咎于 rerank 或额外 CLAHE。尚未逐张审计 GT 四角、cell 图和全部标签。

发布来源 run 本身的训练期验证记录也很差：

- `val_accuracy` 最大值 0.055555556，末值 0.049019609。
- `val_macro_f1` 最大值 0.056896185，末值 0.054495726。
- `val_blank_precision` 最大值 0。
- `score` 最大值 0.045315854，release sidecar 值 0.042775164。

原始记录：`mlruns/245550406113532618/862796eafc5941da9ffa8e43907832aa/metrics/`。这些数字支持“当前 OCR 发布链路必须先审计”，但还不能确定训练数据错位、权重问题或其他训练错误中的哪一个是根因。

### 3. fallback 不是质量 fallback

静态代码事实：`backend/cv/corner_cnn_detector.py::detect` 在成功推理后计算 `last_confidence`，但无论分数如何都返回 `DetectionResult`。`backend/cv/detector.py::ChainDetector.detect` 遇到第一个非 None 就停止。

所以加载成功且正常返回时，即使 Corner 预测错误，OpenCV 也没有机会纠正；当前链主要兜底模型不可用，而不是定位不准。几何分数也不是“图中有数独”的概率。修复方向：增加可拒绝的结构验证，必要时比较两个候选；不能只调几何置信度阈值。

### 4. 训练与评估的可靠性仍不足

- `scripts/train_cnn_ocr.py::build_samples` 用预测角点生成 cell，再直接套原始行列标签：定位误差可造成训练图像/标签错位，应优先使用人工核验的 GT warp。
- 同文件 `stratified_split` 按 cell 切分，会让同一图的 cell 跨训练/验证集。应按原图、同一棋盘或拍摄序列分组。
- `scripts/eval_cnn_ocr.py::load_cells_for_image` 仍把 `detect_grid_image` 的 `(warped, detector_name)` 当图像传给 `prepare_grid_for_ocr`，现有评估入口存在接口漂移；本次实验直接调用各阶段绕开了它。
- Corner 只有 38 张源图；增强不能替代设备、背景、光照、字体和棋盘来源的独立覆盖。

## 推荐路线

### 第一选择：保留混合管线，补几何结构约束

照片 → 粗定位候选 → 原图分辨率网格线精修/验证 → warp → OCR 概率 → 数独约束校验 → 可靠输出或请求重拍/人工确认。

粗定位可继续用现有 Corner CNN，OpenCV 作为对照。精修复用现有 `backend/cv/grid_detector.py` 中的网格线/规则性处理前，需确认其接口与照片场景适用性，而不是复制一套新检测器。结构验证利用两组网格线、交点、投影一致性，不要求强透视原图中的线间距均匀；应在候选校正平面检查近似 9×9 结构。

人工拖四角/纠正少量 OCR 是近期产品兜底，也是故障隔离工具，不是算法已达标。视频场景优先清晰帧选择和稳定后识别，再按实际需要做跨帧融合。

### 第二选择：检测瓶颈确认后，再比较空间定位头

当前 `backend/cv/corner_cnn_arch.py` 为 `forward_features → global average pooling → MLP → 8 coordinates`。它可以学习定位，但先压缩空间维度不是最直接的定位表征。

有条件实验：保留轻量 backbone，比较四角 heatmap（或 soft-argmax/DSNT）与现有坐标头；保持数据切分与训练预算一致。不要同时更换 backbone、输入尺寸、数据与损失，否则不能归因。是否优于当前模型必须以本项目实验确定。

参考原始论文：Nibali et al., *Numerical Coordinate Regression with Convolutional Neural Networks*, https://arxiv.org/abs/1801.07372 。本次网页提取只取得论文书目信息，不据此引用性能数字。

### 第三选择：只有弯曲纸张成为主要失败类时，做局部形变

四角 homography 的前提是平面。书脊/纸张弯曲、显著镜头畸变不能靠更准的四角全部修复；届时才增加内部交点、局部 warp 或去畸变。不要提前升级成 100 点模型。

几何原理来源：OpenCV 官方教程 https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html 。

不建议当前投入：端到端大模型替换整条管线、重新围绕 YOLO 选型、先上可穿戴量化优化。云端多模态可以作为非 camera 路径的离线对照或标注辅助，但须另评估隐私、成本、延迟和整盘 exact match，不能假定输出可解就正确。

## 实施顺序与验收

1. **恢复 OCR（先安排 0.5–1 个工程日定位，不保证当天修好）**：GT warp + 可视化 cell/label；对照历史可用模型、当前 PyTorch checkpoint 与 ONNX；确认输入/类别顺序；发布必须通过固定样例回归。不要先加训练量掩盖错误。
2. **建立分层基线（约 1 个工程日，不含采集标注）**：GT 四角、OpenCV、Corner 三组；照片/截图分开，记录整盘 exact match、给定数字 recall、空白误报和失败阶段。固定独立分组测试集。
3. **定位与拒识（约 1–2 个工程日原型）**：先验证候选网格结构及 fallback 收益；保留识别失败/重拍出口，记录接受覆盖率与接受样本正确率。
4. **数据和模型实验（基线稳定后）**：先补几十张不同来源真实失败照片，明确边框、四角顺序、朝向及裁切标签语义；看学习曲线决定是否扩到数百张。先做粗到精，再有条件比较 heatmap；无准确率提升承诺。

可解率只能作为辅助指标：删错一个 given、读成另一道合法题也可能可解；即便唯一解也不能证明原图识别正确。主指标必须包含原始 givens/blank 的 81 格 exact match，以及相对 GT 的最终答案正确率。IoU 只做定位辅助，另看映射到 cell 尺度的重投影误差与切格质量。

## 复现核心隔离实验

在仓库根目录运行（不训练、不修改模型）：

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import json
import cv2
import numpy as np
from scripts.train_corner_cnn import load_dataset, split_dataset
from backend.cv.detector import warp_grid
from backend.cv.cell_extractor import extract_cells
from backend.api.routes import prepare_grid_for_ocr
from backend.ocr.cnn_digit_reader import CnnDigitReader

samples = load_dataset(Path('data/corner_cnn/labels.csv'), Path('data/corner_cnn/images'))
_, val = split_dataset(samples, 0.2, 42)
labels = json.loads(Path('data/labels/sudoku_labels.json').read_text())
reader = CnnDigitReader(model_path='models/releases/sudoku_digit_cnn_20260402.1.onnx', strict=True)
hit = total = 0
for path, corners in val:
    image = cv2.imread(str(path))
    h, w = image.shape[:2]
    board = warp_grid(image, corners * np.array([w, h]))
    pred = np.array(reader.recognize_grid(extract_cells(prepare_grid_for_ocr(board))))
    truth = np.array(labels[path.name])
    n = int((truth != 0).sum())
    correct = int(((pred == truth) & (truth != 0)).sum())
    hit += correct
    total += n
    print(path.name, correct, '/', n)
print('GT-corner givens recall:', hit, '/', total)
assert hit == total, f'OCR baseline fails even with GT corners: {hit}/{total}'
PY
```

相同处理链已执行，本次观察为 `5 / 222`；末尾断言是后续修复的严格诊断目标，不是声称现有基线通过。该目标只覆盖这 8 张既有图，不能替代独立测试集。
