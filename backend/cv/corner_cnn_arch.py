"""MobileNetV3-Small based corner regression model for sudoku grid detection."""

from __future__ import annotations


def create_corner_model():
    """Create MobileNetV3-Small + regression head for 4-corner prediction.

    Output: 8 values in [0, 1] — (x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl).
    """
    try:
        import timm
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError(
            "timm and PyTorch required. Run: uv sync --extra ml"
        ) from exc

    backbone = timm.create_model(
        "mobilenetv3_small_100",
        pretrained=True,
        num_classes=0,  # remove classifier, return features
    )
    feature_dim = backbone.num_features  # 576 for mobilenetv3_small

    class CornerRegressionCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(64, 8),
                nn.Sigmoid(),
            )

        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)

    return CornerRegressionCNN()
