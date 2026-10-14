"""Corner CNN architectures for sudoku grid detection.

Both take a letterboxed RGB image and output 8 values in [0, 1]:
(x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl) relative to the input.

- ``regress`` (training versions 1.x): MobileNetV3-Small, global average
  pooling and a regression head. Pooling discards where features are, so the
  corners are coarse.
- ``heatmap`` (2.x): the same backbone with a small FPN to stride 4, one
  heatmap per corner, and a soft-argmax that turns each heatmap into
  sub-pixel coordinates.
"""

from __future__ import annotations

ARCHS = ("regress", "heatmap")
FPN_CHANNELS = 64


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
        channels = backbone.feature_info.channels()  # strides 4, 8, 16, 32

        class CornerHeatmapCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.lateral = nn.ModuleList(nn.Conv2d(c, FPN_CHANNELS, 1) for c in channels)
                self.head = nn.Sequential(
                    nn.Conv2d(FPN_CHANNELS, FPN_CHANNELS, 3, padding=1),
                    nn.BatchNorm2d(FPN_CHANNELS),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(FPN_CHANNELS, 4, 1),
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

    raise ValueError(f"Unknown corner model arch: {arch} (expected one of {ARCHS})")
