"""Shared CNN model architecture for Sudoku digit classification."""

from __future__ import annotations


def create_model(arch: str, num_classes: int):
    try:
        import torch
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
        except Exception as exc:
            raise RuntimeError(
                "timm backbone requested but timm is not available"
            ) from exc

    if arch == "custom_small_cnn":

        class SudokuDigitCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Dropout(0.1),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Dropout(0.1),
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

    if arch == "custom_cnn_v3":

        class _SE(nn.Module):
            """Squeeze-and-Excitation: channel attention."""

            def __init__(self, channels: int, reduction: int = 4):
                super().__init__()
                mid = max(channels // reduction, 4)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Sequential(
                    nn.Linear(channels, mid),
                    nn.ReLU(inplace=True),
                    nn.Linear(mid, channels),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                b, c, _, _ = x.shape
                w = self.fc(self.pool(x).view(b, c))
                return x * w.view(b, c, 1, 1)

        class _ResBlock(nn.Module):
            """Two-conv residual block with optional channel projection + SE."""

            def __init__(self, in_ch: int, out_ch: int):
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_ch)
                self.se = _SE(out_ch)
                self.skip = (
                    nn.Conv2d(in_ch, out_ch, 1, bias=False)
                    if in_ch != out_ch
                    else nn.Identity()
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                identity = self.skip(x)
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out = self.se(out)
                return self.relu(out + identity)

        class SudokuDigitCNNv3(nn.Module):
            def __init__(self):
                super().__init__()
                # Stem: 5×5 conv for larger receptive field
                self.stem = nn.Sequential(
                    nn.Conv2d(1, 24, kernel_size=5, padding=2, bias=False),
                    nn.BatchNorm2d(24),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),  # 48→24
                )
                self.block2 = _ResBlock(24, 48)
                self.pool2 = nn.MaxPool2d(2)  # 24→12
                self.block3 = _ResBlock(48, 96)
                self.pool3 = nn.MaxPool2d(2)  # 12→6
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Dropout(0.25),
                    nn.Linear(96, num_classes),
                )

            def forward(self, x):
                x = self.stem(x)
                x = self.pool2(self.block2(x))
                x = self.pool3(self.block3(x))
                return self.head(x)

        return SudokuDigitCNNv3()

    raise ValueError(f"Unknown architecture: {arch}")
