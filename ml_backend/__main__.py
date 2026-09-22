"""Entry point: uv run python -m ml_backend [--port 9090]."""

import argparse

from .app import app


def main() -> None:
    parser = argparse.ArgumentParser(description="Sudoku ML Backend for Label Studio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9090)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
