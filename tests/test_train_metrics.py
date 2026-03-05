"""Tests for compute_metrics extended return values (per_digit_f1, confusion)."""

import numpy as np
import pytest


def _import_torch():
    try:
        import torch
        return torch
    except ImportError:
        pytest.skip("PyTorch not installed")


def test_compute_metrics_returns_extended_keys():
    torch = _import_torch()
    from scripts.train_cnn_ocr import compute_metrics

    # Simulate 20 samples: labels 0..9 repeated twice, perfect predictions.
    labels = torch.tensor([i % 10 for i in range(20)])
    logits = torch.zeros(20, 10)
    for i in range(20):
        logits[i, labels[i]] = 10.0  # high logit for correct class

    result = compute_metrics(logits, labels)

    # Original keys still present.
    assert "accuracy" in result
    assert "macro_f1_digits" in result
    assert "blank_precision" in result

    # New keys.
    assert "per_digit_f1" in result
    assert "confusion" in result

    # per_digit_f1 has keys "1".."9".
    pdf = result["per_digit_f1"]
    assert isinstance(pdf, dict)
    assert set(pdf.keys()) == {str(i) for i in range(1, 10)}
    for v in pdf.values():
        assert 0.0 <= v <= 1.0

    # With perfect predictions all F1 should be 1.0.
    for v in pdf.values():
        assert abs(v - 1.0) < 1e-5

    # Confusion matrix is 10x10.
    cm = result["confusion"]
    assert len(cm) == 10
    assert all(len(row) == 10 for row in cm)

    # Perfect preds → diagonal should match counts.
    for cls in range(10):
        assert cm[cls][cls] == 2  # each class appears twice


def test_compute_metrics_imperfect_predictions():
    torch = _import_torch()
    from scripts.train_cnn_ocr import compute_metrics

    # All predictions are class 1 (everything predicted as "1").
    labels = torch.tensor([0, 1, 2, 3])
    logits = torch.zeros(4, 10)
    logits[:, 1] = 10.0  # always predict 1

    result = compute_metrics(logits, labels)
    assert result["accuracy"] == 0.25  # only label=1 is correct

    # confusion[true_class][1] should be 1 for each true class.
    cm = result["confusion"]
    for cls in range(4):
        assert cm[cls][1] == 1

    # per_digit_f1["1"] should reflect precision=0.25, recall=1.0.
    pdf = result["per_digit_f1"]
    assert pdf["1"] > 0.0
    # Digits 2-9 that appear in labels should have F1=0.
    assert abs(pdf["2"]) < 1e-5
    assert abs(pdf["3"]) < 1e-5
