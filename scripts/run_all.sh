#!/usr/bin/env bash
# Start the FastAPI backend and the Streamlit UI together; Ctrl+C stops both.
#
#   bash scripts/run_all.sh
#   API_PORT=8001 UI_PORT=8502 bash scripts/run_all.sh
#
# Run from anywhere; it switches to the repo root and activates .venv if present.
# The UI reads API_URL from .env (app/config.py); keep it in line with API_PORT.
# If either process exits (e.g. a port is already in use), the other is stopped too.
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"
pids=()

stop_all() {
  trap - INT TERM EXIT
  echo
  echo "Stopping API and UI..."
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap stop_all INT TERM EXIT

uvicorn app.main:app --port "$API_PORT" &
pids+=("$!")

# .streamlit/config.toml: localhost only, headless, no usage statistics.
streamlit run ui/streamlit_app.py --server.port "$UI_PORT" &
pids+=("$!")

echo "API: http://localhost:${API_PORT}/docs"
echo "UI:  http://localhost:${UI_PORT}"
echo "Press Ctrl+C to stop both."

# Return as soon as either process exits; the EXIT trap then stops the other.
wait -n
