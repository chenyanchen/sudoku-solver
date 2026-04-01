"""Export corner regression CNN checkpoint to ONNX format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.cv.corner_cnn_arch import create_corner_model


def parse_args():
    parser = argparse.ArgumentParser(description="Export corner CNN to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True, help="PyTorch .pt path")
    parser.add_argument("--output", type=Path, required=True, help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    payload = torch.load(args.checkpoint, map_location="cpu")
    input_size = payload.get("input_size", 256)

    model = create_corner_model()
    model.load_state_dict(payload["state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, input_size, input_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        input_names=["input"],
        output_names=["corners"],
        dynamic_axes={"input": {0: "batch"}, "corners": {0: "batch"}},
        opset_version=args.opset,
    )

    print(f"Exported ONNX: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
