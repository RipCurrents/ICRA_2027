#!/usr/bin/env bash
# One-command start of the rip-current edge app.
#   ./run.sh                     UI at http://<host>:7860  (RIPAPP_PORT overrides the port)
#   ./run.sh headless [jobs.json] headless batch over HEADLESS_JOBS in app.py, or over the json list given
# Python: $RIPAPP_PYTHON if set, else the icra_app conda env, else python3 on PATH (needs requirements.txt installed).
# Paths on another machine (Jetson): export RIPAPP_EXPORTS_DIR / RIPAPP_OUT_ROOT / RIPAPP_ENGINE_DIR before starting
# (see README.md); everything else is a constant at the top of app.py.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${RIPAPP_PYTHON:-}"
if [ -z "$PY" ]; then
  for c in "$HOME/anaconda3/envs/icra_app/bin/python" "$HOME/miniconda3/envs/icra_app/bin/python" "$(command -v python3 || true)"; do
    [ -n "$c" ] && [ -x "$c" ] && PY="$c" && break
  done
fi
[ -n "$PY" ] || { echo "no python found; set RIPAPP_PYTHON"; exit 1; }
if [ "${1:-ui}" = "headless" ]; then
  export RIPAPP_MODE=headless
  [ -n "${2:-}" ] && export RIPAPP_JOBS="$2"
  "$PY" -c "import onnxruntime, cv2, numpy" 2>/dev/null || { echo "missing deps: $PY -m pip install -r $HERE/requirements.txt"; exit 1; }
else
  "$PY" -c "import gradio, onnxruntime, cv2, numpy" 2>/dev/null || { echo "missing deps: $PY -m pip install -r $HERE/requirements.txt"; exit 1; }
fi
mkdir -p "${RIPAPP_OUT_ROOT:-/mnt/linux/icra_edge/app_runs}" 2>/dev/null || true
exec "$PY" "$HERE/app.py"
