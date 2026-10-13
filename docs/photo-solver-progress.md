# 普通照片求解：修复与验收

## 范围

当前回归目标是 `IMG_8695.jpg`–`IMG_8708.jpg`：14 张普通数独屏幕的实拍照片，包含透视、反光和摩尔纹。`IMG_8709.jpg`–`IMG_8712.jpg` 是装饰边框照片，单独记录，不要求本轮通过。

这是已知图片回归，不是任意相机、纸质数独、弯曲纸张的泛化证明。下一轮需要从不同设备/场景采集未参与训练或调参的新照片。所有照片标注均由助手逐格目视复核；不是由求解器或 CNN 输出反填标签。应保留人工再次复核的机会。

## 已定位并修复的问题

1. **OCR 训练裁切错误**：旧 release 来源 run 使用 `--use-yolo=True`。从 Git 恢复的历史 YOLO ONNX 输出 `[1,300,38]`，元数据 `kpt_shape=[16,2]`；训练代码默认按 4 点读取，取到了同一顶边上的四个点。以 `sudoku_5.jpg` 为例，错误四角面积约 2,886 px²，正确外角面积约 2,262,328 px²。新 OCR 训练只使用版本控制内的 GT 四角，不再走 YOLO 或 Corner 推断裁切。
2. **数字标注错位/错误**：18 张照片中修改了 17 张数字标注，1 张确认无误。`IMG_8700.jpg` 原图 r1 为 `300049000`，旧 JSON 为 `020910500`，显然不是同一棋盘。修订后 18 张标签均无冲突且有唯一解；唯一解只是辅助校验，不是标签的来源。`data/labels/sudoku_label_sources.json` 记录复核来源。
3. **四角标注错误**：修正 `IMG_8698.jpg`。OpenCV 候选经目视检查与原图网格一致后写入 `data/labels/sudoku_corners.csv`；其余角点从已有标注迁入。OCR 训练/评估不再依赖 gitignored 的 `data/corner_cnn/images/` 副本。
4. **照片预处理和阈值**：原有 CLAHE + cell Otsu 会把缓慢光照变化放大成黑块。现在先在整个棋盘上做局部阈值，再切格。低于数字阈值的预测不能在二次识别失败后被无条件保留为 given；单格与批量共用选择逻辑。
5. **验收和发布**：Corner 输出须通过已有网格线规则性检查，否则允许 OpenCV 接管；OCR 按棋盘分组切分，不拆散同图或重复 givens；最佳 checkpoint 深拷贝，防止被后续训练覆盖；导出验证 PyTorch/ONNX 数值一致；发布必须通过 14 张普通照片及 2 张截图的 exact-match 门禁，失败不更换 canonical。

## 对上一份报告的修正

`visual-sudoku-route-review.md` 中的 `5/222` 是对旧错误标签的比较值，**不能作为真实照片 OCR 准确率**。保留该记录用于追踪诊断过程，不作为新旧模型收益的分母。

旧 OCR 发布的训练日志低分及错误裁切仍是可独立复现的事实；Corner `0.5916` 仍是旧四角标签下的 mask IoU，不是整盘准确率。旧 Corner 训练标签修订后不能直接横向比较 IoU。

## 已发布模型结果

Release **`20260922.2`** 已切换为 `models/releases/sudoku_digit_cnn.onnx` 的目标。训练版本 `3.5.2`，run `2676e889344242229186ae6ee73620fd`（在修复工作树上训练；sidecar 的 `source_git_commit=90dcddd` 是当时的基底提交，不包含未提交修复）：

| 检查 | 结果 |
|---|---|
| 普通照片，自动检测链 → OCR → solver | 14/14 整盘 exact，14/14 最终答案 exact |
| 普通照片 givens / blank | 425/425 给定数字正确，空白误报 0 |
| 38 张混合数据，默认自动检测链 | 31/38 整盘 exact，31/38 最终答案 exact |
| 定向测试（含 Web 上传识别、JSON 求解） | 64 passed |

OCR 分组验证集为 `sudoku_3.jpg`, `sudoku_4.jpg`, `sudoku_6.jpg`, `IMG_8697.jpg`, `IMG_8708.jpg`, `IMG_8709.jpg`, `IMG_8712.jpg`。它用于训练选优，并已用于诊断/调参，不能冒充独立测试集。14 张普通照片中只有 `IMG_8697.jpg`、`IMG_8708.jpg` 未参与该次训练。

剩余不通过图片：`sudoku_8.jpg`、`sudoku_10.jpeg`、`sudoku_21.jpg` 及 4 张装饰边框照片。前两张截图还需再次核对数字标注与图像；本轮没有为了提高指标修改它们。

## 复现

普通照片严格验收（必须同时整盘识别正确、求解答案正确）：

```bash
uv run python scripts/eval_cnn_ocr.py \
  --cnn-model models/releases/sudoku_digit_cnn.onnx \
  --detector auto --strict \
  --image-pattern 'IMG_86*.jpg' 'IMG_870[0-8].jpg'
```

Web 真实上传 → 识别 → 求解回归：

```bash
uv run --extra dev pytest tests/test_photo_solver.py -q
```

训练（自动记录数据哈希和 `split.json`）：

```bash
MLFLOW_TRACKING_URI="file://$PWD/mlruns" \
  uv run --extra ml python scripts/train_cnn_ocr.py \
  --version 3.5.2 --epochs 60 --patience 20
```

发布时仍需显式指定训练 run：

```bash
MLFLOW_TRACKING_URI="file://$PWD/mlruns" \
  uv run --extra ml python scripts/release_model.py cnn \
  --run-id <VERIFIED_RUN_ID> --version <YYYYMMDD.N>
```

发布元数据保留类别顺序、输入尺寸、训练版本和预处理版本，OCR reader 会解析 canonical symlink 邻接的真实 sidecar。

### 被拒绝的重训候选

修复提交 `f133269` 后，使用相同参数重训得到 run `820870236327483bb34e1b78fcad2369`。尽管验证 score 仍为 1.0，实际自动定位照片回归中 `IMG_8697.jpg` 和 `IMG_8705.jpg` 未整盘正确（12/14 照片通过）。发布门禁拒绝候选 `20260922.1`，没有更换默认权重。该未发布候选归档在 gitignored 的 `tmp/photo-diagnosis/`。

这表明当前训练选优指标仍不能替代生产路径回归，固定 seed 也不保证此次运行环境中的权重完全一致。本轮保留实际通过回归的既有候选，未降低门禁阈值。正式 `20260922.2` 发布后已重新执行 64 项定向测试，全部通过。

## 测试范围与未完成项

定向测试覆盖本轮修改涉及的 CV、OCR、solver、训练数据、发布门禁和 API。完整 `pytest` 收集仍被两个已有问题阻断：`scripts/test_yolo_photo.py` 的 `screen_monitor` 模块遮蔽，以及 `tests/test_train_yolo_grid.py` 引用已删除的训练模块。本轮不宣称全库测试通过。

下一步：保持普通照片回归通过，新增独立设备/纸质照片评估；然后处理真实剩余失败。动画边框继续不作为验收条件。不要因为当前 14/14 就继续加网络层或对外宣称通用准确率 100%。

## Release 20261013.1

训练版本 `3.6.0`，run `6669cc1d00c346fcb01fc25870ebe69b`。和 `20260922.2` 相比只有两处变化：

- **修正了 3 格标注。** `sudoku_8` 第 2 行第 4 列应为 1；`sudoku_10` 第 6 行第 4 列应为空格，第 8 行第 6 列应为 3。这几处错误是新求解器检查出来的：原标注无解。
- **修复了 checkpoint 选择。** 验证集在第 8 轮就达到 score 1.0，旧逻辑只在分数严格提高时更新，所以一直保存第 8 轮的权重，并在第 28 轮早停。现在分数持平时取更靠后的 checkpoint，这次训满了 60 轮。

先训的那个候选（停在第 8 轮）被门禁拒绝了：原始识别 29/38，空格误报 22。

| 检查 | 20260922.2 | 20261013.1 |
|---|---|---|
| 14 张普通照片，经过 decode 的完整流程（`--strict`） | 14/14 | 14/14 |
| 38 张标注图，原始识别 / 完整流程 整盘正确 | 32 / 33 | 32 / 33 |
| 空格误识别为数字 | 14 | 11 |
| 40 张图（含两张截图）完整流程，错误解 | 35/40，0 | 35/40，0 |
| `pytest` | 121 passed | 121 passed |
