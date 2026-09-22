from __future__ import annotations

import re
from pathlib import Path
from typing import Any

CORNER_LABEL_ORDER = ("TL", "TR", "BR", "BL")


def is_keypoints_project(parsed_label_config: dict[str, Any]) -> bool:
    return any(schema.get("type") == "KeyPointLabels" for schema in parsed_label_config.values())


def _find_schema(
    parsed_label_config: dict[str, Any], expected_type: str
) -> tuple[str, dict[str, Any]]:
    for from_name, schema in parsed_label_config.items():
        if schema.get("type") == expected_type:
            return from_name, schema
    raise KeyError(f"No {expected_type} tag found in parsed label config")


def get_image_data_key(schema: dict[str, Any]) -> str:
    inputs = schema.get("inputs", [])
    if not inputs:
        return "image"
    return inputs[0].get("value", "image")


def extract_keypoint_binding(parsed_label_config: dict[str, Any]) -> tuple[str, str, list[str]]:
    from_name, schema = _find_schema(parsed_label_config, "KeyPointLabels")
    return from_name, schema["to_name"][0], list(schema.get("labels", []))


def ordered_corner_labels(parsed_label_config: dict[str, Any]) -> list[str]:
    _from_name, _to_name, labels = extract_keypoint_binding(parsed_label_config)
    label_set = set(labels)
    if label_set.issuperset(CORNER_LABEL_ORDER):
        return list(CORNER_LABEL_ORDER)
    return labels


def extract_textarea_binding(parsed_label_config: dict[str, Any]) -> tuple[str, str]:
    from_name, schema = _find_schema(parsed_label_config, "TextArea")
    return from_name, schema["to_name"][0]


def extract_image_ref(task: dict[str, Any], parsed_label_config: dict[str, Any]) -> str:
    data = task.get("data", {})
    try:
        if is_keypoints_project(parsed_label_config):
            _from_name, _to_name, _labels = extract_keypoint_binding(parsed_label_config)
            key = get_image_data_key(parsed_label_config[_from_name])
        else:
            from_name, _to_name = extract_textarea_binding(parsed_label_config)
            key = get_image_data_key(parsed_label_config[from_name])
    except KeyError:
        key = "image"

    image_ref = data.get(key, "")
    if image_ref:
        return image_ref

    for fallback_key in ("image", "img"):
        image_ref = data.get(fallback_key, "")
        if image_ref:
            return image_ref
    return ""


def extract_filename(
    task: dict[str, Any],
    parsed_label_config: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> str:
    image_ref = extract_image_ref(task, parsed_label_config or {})
    basename = image_ref.rstrip("/").split("/")[-1]
    match = re.match(r"^[a-f0-9]+-(.+)$", basename)
    if match:
        return match.group(1)

    if repo_root is not None:
        stem = Path(basename).stem.removesuffix("_warped")
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = repo_root / "data" / "raw" / "images" / f"{stem}{ext}"
            if candidate.exists():
                return f"{stem}{ext}"

    return basename
