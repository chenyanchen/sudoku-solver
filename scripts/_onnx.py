"""ONNX file helpers shared by the export scripts."""

from __future__ import annotations

from pathlib import Path

import onnx

STACK_TRACE_KEY = "pkg.torch.onnx.stack_trace"


def strip_stack_traces(path: Path) -> None:
    """Drop the Python stack traces torch.onnx.export records on each node.

    They hold absolute paths of the exporting machine and are not used at
    inference. External weights are left untouched.
    """
    model = onnx.load(path, load_external_data=False)
    for node in model.graph.node:
        kept = [prop for prop in node.metadata_props if prop.key != STACK_TRACE_KEY]
        del node.metadata_props[:]
        node.metadata_props.extend(kept)
    onnx.save_model(model, path)
