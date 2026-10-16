from __future__ import annotations

import pytest
import torch

from backend.cv.corner_cnn_arch import ARCHS, create_corner_model, soft_argmax


@pytest.mark.parametrize("arch", ARCHS)
def test_model_outputs_8_corner_values_in_unit_range(arch):
    model = create_corner_model(arch, pretrained=False).eval()

    with torch.no_grad():
        outputs = model(torch.randn(2, 3, 256, 256))

    assert outputs.shape == (2, 8)
    assert ((outputs >= 0) & (outputs <= 1)).all()


def test_soft_argmax_finds_a_sharp_peak():
    logits = torch.full((1, 1, 64, 64), -50.0)
    logits[0, 0, 16, 48] = 50.0  # row 16, column 48

    coords, prob = soft_argmax(logits)

    assert torch.allclose(coords[0, 0], torch.tensor([48.5 / 64, 16.5 / 64]), atol=1e-4)
    assert torch.isclose(prob.sum(), torch.tensor(1.0))


def test_tiny_model_takes_any_multiple_of_32():
    model = create_corner_model("tiny_16-32-64-128").eval()

    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 384, 384))

    assert outputs.shape == (1, 8)


@pytest.mark.parametrize("arch", ["tiny_16-32-64", "tiny_16-32-64-128_wide"])
def test_tiny_arch_needs_four_widths_and_a_known_suffix(arch):
    with pytest.raises(ValueError, match="expected tiny_"):
        create_corner_model(arch)
