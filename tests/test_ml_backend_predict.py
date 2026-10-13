from __future__ import annotations


from ml_backend.predictors import decode_image
from ml_backend.schema import extract_image_ref


def test_decode_image_returns_none_for_empty_ref_without_imread(monkeypatch):
    def fail_imread(path: str):
        raise AssertionError(f"cv2.imread should not be called for empty path: {path!r}")

    import ml_backend.predictors as predictors

    monkeypatch.setattr(predictors.cv2, "imread", fail_imread)

    assert decode_image("") is None


def test_extract_image_ref_uses_label_config_value_binding():
    parsed_label_config = {
        "kp-1": {
            "type": "KeyPointLabels",
            "to_name": ["img-1"],
            "inputs": [{"type": "Image", "value": "img"}],
            "labels": ["TL"],
            "labels_attrs": {},
        }
    }
    task = {"data": {"img": "/data/upload/6/example.jpg"}}

    assert extract_image_ref(task, parsed_label_config) == "/data/upload/6/example.jpg"


def test_extract_image_ref_falls_back_to_img_when_schema_missing():
    task = {"data": {"img": "/data/upload/6/example.jpg"}}

    assert extract_image_ref(task, {}) == "/data/upload/6/example.jpg"
