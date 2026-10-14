"""Training/export tests need the `ml` extra; skip them in a `dev`-only environment."""

from importlib.util import find_spec

_ML_TESTS = [
    "test_corner_cnn_arch.py",
    "test_export_corner_cnn_onnx.py",
    "test_ocr_training_data.py",
    "test_release_model.py",
    "test_train_metrics.py",
]

collect_ignore = [] if all(find_spec(m) for m in ("torch", "timm", "onnx", "mlflow")) else _ML_TESTS
