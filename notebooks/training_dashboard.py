"""Training metrics dashboard — run with: marimo run notebooks/training_dashboard.py"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def cell_file_input():
    import marimo as mo

    metrics_path = mo.ui.text(
        label="Metrics JSONL path",
        value="models/checkpoints/sudoku_digit_cnn.metrics.jsonl",
    )
    metrics_path
    return (metrics_path,)


@app.cell
def cell_load_data(metrics_path):
    import json
    from pathlib import Path

    import marimo as mo

    path = Path(metrics_path.value)
    records = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    mo.md(f"**Loaded {len(records)} epoch records** from `{path}`")
    return (records,)


@app.cell
def cell_loss_curves(records):
    import matplotlib.pyplot as plt
    import marimo as mo

    if not records:
        mo.md("_No data to plot._")
        return

    epochs = [r["epoch"] for r in records]
    train_loss = [r["train_loss"] for r in records]
    val_loss = [r["val_loss"] for r in records]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_loss, label="Train Loss", marker=".")
    ax.plot(epochs, val_loss, label="Val Loss", marker=".")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig
    return


@app.cell
def cell_accuracy_f1(records):
    import matplotlib.pyplot as plt
    import marimo as mo

    if not records:
        mo.md("_No data to plot._")
        return

    epochs = [r["epoch"] for r in records]
    acc = [r["val_accuracy"] for r in records]
    f1 = [r["val_macro_f1_digits"] for r in records]
    score = [r["score"] for r in records]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, acc, label="Accuracy", marker=".")
    ax.plot(epochs, f1, label="Macro F1 (digits)", marker=".")
    ax.plot(epochs, score, label="Score", marker=".", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("Accuracy / F1 / Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig
    return


@app.cell
def cell_per_digit_f1(records):
    import matplotlib.pyplot as plt
    import numpy as np
    import marimo as mo

    if not records or "per_digit_f1" not in records[-1]:
        mo.md("_No per-digit F1 data._")
        return

    # Build a matrix: rows = epochs, cols = digits 1..9.
    digits = [str(d) for d in range(1, 10)]
    matrix = []
    epoch_labels = []
    for r in records:
        pdf = r.get("per_digit_f1", {})
        row = [pdf.get(d, 0.0) for d in digits]
        matrix.append(row)
        epoch_labels.append(r["epoch"])

    arr = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, max(3, len(records) * 0.25)))
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(digits)))
    ax.set_xticklabels(digits)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Epoch")
    ax.set_yticks(range(len(epoch_labels)))
    ax.set_yticklabels(epoch_labels)
    ax.set_title("Per-Digit F1 Heatmap")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig
    return


@app.cell
def cell_confusion_matrix(records):
    import matplotlib.pyplot as plt
    import numpy as np
    import marimo as mo

    if not records or "confusion" not in records[-1]:
        mo.md("_No confusion matrix data._")
        return

    cm = np.array(records[-1]["confusion"])
    labels = [str(i) for i in range(10)]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(10))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (Epoch {records[-1]['epoch']})")

    # Annotate cells.
    for i in range(10):
        for j in range(10):
            val = int(cm[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
