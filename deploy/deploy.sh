#!/usr/bin/env bash
# Deploy the committed HEAD: deploy.sh <ssh-host>
# The server gets exactly a commit (git archive), in its own release directory;
# rolling back is pointing `current` at a previous release and restarting.
set -euo pipefail
HOST=${1:?usage: deploy.sh <ssh-host>}
cd "$(git rev-parse --show-toplevel)"
git diff --quiet HEAD -- backend frontend models pyproject.toml uv.lock \
    || { echo "Uncommitted changes in deployed files; commit first." >&2; exit 1; }
REV=$(git rev-parse --short HEAD)
DIR=/opt/sudoku-solver/releases/$REV

REQS=$(mktemp)
trap 'rm -f "$REQS"' EXIT
uv export --frozen --no-dev --no-hashes --no-emit-project > "$REQS"
ssh "$HOST" "mkdir -p $DIR"
git archive HEAD backend frontend models/releases | ssh "$HOST" "tar -x -C $DIR"
scp -q "$REQS" "$HOST:$DIR/requirements.txt"
ssh "$HOST" bash -s <<REMOTE
set -euo pipefail
cd /opt/sudoku-solver
venv/bin/pip install -q --disable-pip-version-check -r $DIR/requirements.txt
ln -sfn $DIR current
systemctl restart sudoku-solver
for i in \$(seq 1 30); do curl -fs -o /dev/null http://127.0.0.1:8000/health && break; sleep 1; done
curl -fsS http://127.0.0.1:8000/health; echo
# Read every kept upload again with this release, so the public solve rate is
# what it solves (backend/rescore.py). Lowest CPU priority, beside the live
# app; follow it with: journalctl -u 'sudoku-rescore-*'
systemd-run --quiet --collect --unit=sudoku-rescore-$REV-\$(date +%s) --uid=sudoku --gid=sudoku \
    --nice=19 -p MemoryMax=1G -p ReadWritePaths=/var/lib/sudoku-solver --working-directory=$DIR \
    -E CNN_MODEL_PATH=$DIR/models/releases/sudoku_digit_cnn.onnx -E SUDOKU_DATA_DIR=/var/lib/sudoku-solver \
    -E OPENCV_IO_MAX_IMAGE_PIXELS=50000000 \
    /opt/sudoku-solver/venv/bin/python -m backend.rescore --version $REV
echo "Rescoring uploads in the background (journalctl -u 'sudoku-rescore-*')"
# Keep the three newest releases, and whichever one is current.
ls -1dt releases/* | tail -n +4 | grep -vx "releases/$REV" | xargs -r rm -rf
REMOTE
echo "Deployed $REV to $HOST"
