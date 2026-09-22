from __future__ import annotations

import torch

from backend.cv.corner_cnn_arch import create_corner_model


def test_create_corner_model_forward_returns_8_corner_values():
    model = create_corner_model()
    inputs = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (2, 8)
