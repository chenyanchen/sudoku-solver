# Detector Abstraction — Design Spec

将检测逻辑抽象为统一接口，解耦检测策略与消费者。

## 接口定义

```python
@dataclass
class DetectionResult:
    corners: np.ndarray  # (4, 2), [TL, TR, BR, BL] in original image coords

class GridDetector(Protocol):
    def detect(self, image: np.ndarray) -> DetectionResult | None: ...
```

`DetectionResult` 只包含 corners。confidence 等模型特有属性留在具体实现内部，不属于「检测棋盘」的本质行为。

warp 是纯几何操作（`getPerspectiveTransform + warpPerspective`），不依赖任何检测器，作为独立工具函数：

```python
def warp_grid(image: np.ndarray, corners: np.ndarray, size: int = 450) -> np.ndarray:
    """Warp grid region to a square image using 4 corner coordinates."""
```

## 三个实现

### YoloGridDetector

封装现有 YOLO26n-pose 推理。内部持有 16-kpt keypoints 和 confidence，但 `detect()` 只返回 4 角 corners。消费者如需 confidence 可通过具体类型访问（如 `detector.last_confidence`）。

### OpenCVGridDetector

封装现有 `grid_detector.py` 的轮廓检测 + 规整性打分 + 线性回归细化逻辑。无 confidence 概念。

### ChainDetector

组合多个 `GridDetector`，按优先级依次尝试，第一个返回非 None 结果即采用。

```python
class ChainDetector:
    def __init__(self, detectors: list[GridDetector]) -> None: ...
    def detect(self, image: np.ndarray) -> DetectionResult | None:
        for d in self.detectors:
            result = d.detect(image)
            if result is not None:
                return result
        return None
```

## 消费者变更

### Web App (`backend/api/routes.py`)

当前 `detect_grid_image()` 硬编码 YOLO→OpenCV→CLAHE retry 链。重构为：

```python
_detector = ChainDetector([YoloGridDetector(...), OpenCVGridDetector()])

def detect_grid_image(image: np.ndarray) -> np.ndarray | None:
    result = _detector.detect(image)
    if result is None:
        return None
    return warp_grid(image, result.corners)
```

CLAHE retry 逻辑移入 `OpenCVGridDetector` 内部（它本身就有多阶段检测策略）。

### Screen Monitor (`screen_monitor/`)

当前直接调用 `find_grid_with_corners()`。改为使用 `GridDetector` 接口，初始化时传入具体实现。

### Camera Monitor (`camera_monitor/`)

当前直接用 `YoloGridDetector`。改为使用 `GridDetector` 接口，不需要 `ChainDetector`（相机场景只用 YOLO）。

## 文件结构

| Action | File | 说明 |
|--------|------|------|
| Create | `backend/cv/detector.py` | `DetectionResult`、`GridDetector` Protocol、`warp_grid()`、`ChainDetector` |
| Modify | `backend/cv/yolo_detector.py` | `detect()` 返回 `DetectionResult`，warp 移出 |
| Create | `backend/cv/opencv_detector.py` | 封装现有 `grid_detector.py` 逻辑为 `OpenCVGridDetector` |
| Modify | `backend/api/routes.py` | 用 `ChainDetector` 替代硬编码逻辑 |
| Modify | `screen_monitor/agent.py` 或相关文件 | 切换到 `GridDetector` 接口 |
| Modify | `camera_monitor/app.py` | 切换到 `GridDetector` 接口 |
