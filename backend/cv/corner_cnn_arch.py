"""Corner CNN architectures for sudoku grid detection.

Both take a letterboxed RGB image and output 8 values in [0, 1]:
(x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl) relative to the input.

- ``regress`` (training versions 1.x): MobileNetV3-Small, global average
  pooling and a regression head. Pooling discards where features are, so the
  corners are coarse.
- ``heatmap`` (2.x): the same backbone with a small FPN to stride 4, one
  heatmap per corner, and a soft-argmax that turns each heatmap into
  sub-pixel coordinates.
- ``tiny_<w4>-<w8>-<w16>-<w32>`` (3.x): the heatmap head on a small
  depthwise-separable backbone trained from scratch, with the given widths at
  strides 4 to 32 (two convs each, after a stride-2 stem to 16 channels).
  Stride 32 lets every corner see the whole grid: locally an outer corner
  looks like a box corner inside it.
"""

from __future__ import annotations

ARCHS = ("regress", "heatmap", "tiny_16-32-64-128")
FPN_CHANNELS = 64
TINY_FPN_CHANNELS = 32


def _import():
    try:
        import timm
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("timm and PyTorch required. Run: uv sync --extra ml") from exc
    return timm, torch, nn, F


def soft_argmax(logits):
    """(N, K, H, W) logits -> ((N, K, 2) coordinates in [0, 1], (N, K, H, W) probabilities)."""
    _, torch, _, F = _import()
    n, k, h, w = logits.shape
    prob = F.softmax(logits.reshape(n, k, h * w), dim=-1).reshape(n, k, h, w)
    xs = (torch.arange(w, dtype=prob.dtype, device=prob.device) + 0.5) / w
    ys = (torch.arange(h, dtype=prob.dtype, device=prob.device) + 0.5) / h
    x = (prob.sum(dim=2) * xs).sum(dim=-1)
    y = (prob.sum(dim=3) * ys).sum(dim=-1)
    return torch.stack([x, y], dim=-1), prob


def create_corner_model(arch: str = "regress", pretrained: bool = True):
    timm, torch, nn, F = _import()
    if arch == "regress":
        backbone = timm.create_model("mobilenetv3_small_100", pretrained=pretrained, num_classes=0)
        feature_dim = backbone.num_features

        class CornerRegressionCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.head = nn.Sequential(
                    nn.Linear(feature_dim, 64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(64, 8),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                features = self.backbone.forward_features(x)
                features = self.pool(features).flatten(1)
                return self.head(features)

        return CornerRegressionCNN()

    if arch == "heatmap":
        backbone = timm.create_model(
            "mobilenetv3_small_100", pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4),
        )
        return _heatmap_model(backbone, backbone.feature_info.channels(), FPN_CHANNELS)

    if arch.startswith("tiny_"):
        widths = [int(w) for w in arch[len("tiny_"):].split("-")]
        if len(widths) != 4:
            raise ValueError(f"{arch}: expected four widths, for strides 4, 8, 16 and 32")
        return _heatmap_model(_tiny_backbone(widths), widths, TINY_FPN_CHANNELS)

    raise ValueError(f"Unknown corner model arch: {arch} (expected one of {ARCHS})")


def _tiny_backbone(widths: list[int]):
    """Features at strides 4, 8, 16 and 32, as the timm backbone returns them."""
    _, _, nn, _ = _import()
    from backend.ocr.model_arch import _conv_bn_relu

    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(*_conv_bn_relu(3, 16, separable=False, stride=2))
            stages, in_ch = [], 16
            for width in widths:
                stages.append(nn.Sequential(
                    *_conv_bn_relu(in_ch, width, separable=True, stride=2),
                    *_conv_bn_relu(width, width, separable=True),
                ))
                in_ch = width
            self.stages = nn.ModuleList(stages)

        def forward(self, x):
            features, x = [], self.stem(x)
            for stage in self.stages:
                x = stage(x)
                features.append(x)
            return features

    return TinyBackbone()


def _heatmap_model(backbone, channels, fpn_channels: int):
    """An FPN from `backbone`'s strides 4-32 features to one heatmap per corner at stride 4."""
    _, _, nn, F = _import()

    class CornerHeatmapCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.lateral = nn.ModuleList(nn.Conv2d(c, fpn_channels, 1) for c in channels)
            self.head = nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(fpn_channels, 4, 1),
            )

        def heatmaps(self, x):
            """Per-corner logits at stride 4."""
            features = self.backbone(x)
            y = self.lateral[-1](features[-1])
            for feature, lateral in zip(reversed(features[:-1]), reversed(self.lateral[:-1])):
                y = F.interpolate(y, size=feature.shape[-2:], mode="bilinear", align_corners=False)
                y = y + lateral(feature)
            return self.head(y)

        def forward(self, x):
            coords, _ = soft_argmax(self.heatmaps(x))
            return coords.flatten(1)

    return CornerHeatmapCNN()
