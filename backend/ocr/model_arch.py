"""Shared CNN model architecture for Sudoku digit classification."""

from __future__ import annotations


def create_model(arch: str, num_classes: int):
    try:
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required. Install ML stack first: uv sync --extra ml"
        ) from exc

    if arch == "timm_mobilenetv3":
        try:
            import timm

            return timm.create_model(
                "mobilenetv3_small_100",
                pretrained=False,
                in_chans=1,
                num_classes=num_classes,
            )
        except Exception:
            print("[WARN] timm unavailable, fallback to custom_small_cnn")

    class SudokuDigitCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.1),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.1),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(64, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    return SudokuDigitCNN()
