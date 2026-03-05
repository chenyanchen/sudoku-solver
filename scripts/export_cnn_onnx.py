"""Export trained Sudoku CNN checkpoint to ONNX format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.ocr.model_arch import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Sudoku CNN to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "models" / "checkpoints" / "sudoku_digit_cnn.pt",
        help="PyTorch checkpoint path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "models" / "releases" / "sudoku_digit_cnn_v2.0.onnx",
        help="Output ONNX path",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply INT8 dynamic quantization post-export",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    payload = torch.load(args.checkpoint, map_location="cpu")
    state_dict = payload["state_dict"]
    class_values = payload.get("class_values", list(range(10)))
    input_size = int(payload.get("input_size", 48))
    arch = payload.get("arch", "custom_small_cnn")
    version = payload.get("version", "1.0.0")

    model = create_model(arch, num_classes=len(class_values))
    model.load_state_dict(state_dict)
    model.eval()

    dummy = torch.randn(1, 1, input_size, input_size, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )

    if args.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantized_path = args.output.with_stem(args.output.stem + "_int8")
        quantize_dynamic(
            str(args.output),
            str(quantized_path),
            weight_type=QuantType.QInt8,
        )
        print(f"Quantized ONNX (INT8): {quantized_path}")

    sidecar = {
        "version": version,
        "class_values": class_values,
        "input_size": input_size,
        "arch": arch,
    }
    sidecar_path = args.output.with_suffix(".meta.json")
    with sidecar_path.open("w", encoding="utf-8") as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=2)

    print(f"Exported ONNX: {args.output}")
    print(f"Exported metadata: {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
