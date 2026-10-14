"""Export corner regression CNN checkpoint to ONNX format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.cv.corner_cnn_arch import create_corner_model


def parse_args():
    parser = argparse.ArgumentParser(description="Export corner CNN to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True, help="PyTorch .pt path")
    parser.add_argument("--output", type=Path, required=True, help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def _load_onnx_opset(path: Path) -> int:
    model = onnx.load(path)
    for opset in model.opset_import:
        if opset.domain == "":
            return int(opset.version)
    raise RuntimeError(f"No default ONNX opset import found in {path}")


def export_checkpoint(args: argparse.Namespace) -> Path:
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    payload = torch.load(args.checkpoint, map_location="cpu")
    input_size = payload.get("input_size", 256)

    model = create_corner_model(payload.get("arch", "regress"), pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, input_size, input_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    batch_dim = torch.export.Dim("batch")

    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        input_names=["input"],
        output_names=["corners"],
        dynamic_shapes=({0: batch_dim},),
        dynamic_axes=None,
        opset_version=args.opset,
    )

    actual_opset = _load_onnx_opset(args.output)
    if actual_opset != args.opset:
        raise RuntimeError(f"Expected ONNX opset {args.opset}, got {actual_opset}")

    return args.output


def main() -> int:
    args = parse_args()
    output = export_checkpoint(args)

    print(f"Exported ONNX: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
