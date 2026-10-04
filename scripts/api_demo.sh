#!/usr/bin/env bash
# Demo of the /chat API (spec §1 success criteria, §9). Needs the server running:
#   uvicorn app.main:app
# Then, from the repo root:
#   bash scripts/api_demo.sh                 # against http://localhost:8000
#   BASE_URL=http://host:port bash scripts/api_demo.sh
# Requires curl and jq. Every question uses as_of 2026-10-03 (spec §8.4 date).
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
AS_OF="2026-10-03"

# ask USER_ID QUESTION: POST /chat as that user, print answer, citations, tools, latency, audit id.
ask() {
  local user="$1" question="$2" headers body
  headers="$(mktemp)"
  body="$(jq -n --arg m "$question" --arg d "$AS_OF" '{message: $m, as_of: $d}')"
  echo "=== ${user}: ${question}"
  curl -sS -D "$headers" -X POST "$BASE_URL/chat" \
    -H "X-User-Id: ${user}" -H "Content-Type: application/json" -d "$body" |
    jq -r '"answer: \(.answer // .detail)",
           (.citations // [] | .[] | "  cite: [\(.doc_title), p. \(.page)] \(.section)"),
           (.tool_calls // [] | .[] | "  tool: \(.name) \(.args | tojson)"),
           "  latency_ms: \(.latency_ms // "-")"'
  echo "  audit id: $(grep -i '^x-audit-id:' "$headers" | cut -d' ' -f2 | tr -d '\r')"
  rm -f "$headers"
  echo
}

ask u_priya "What happens to my unvested options if I resign?"
ask u_priya "How many options have I vested as of today?"
ask u_priya "If I leave next month, how many options do I keep and how long do I have to exercise them?"
ask u_priya "Show me Rahul's grant."
ask u_arjun "If we issue 2,000,000 new shares to a new investor, Horizon Capital, how does my ownership change?"

echo "=== Access checks (expect 403, 403, 401)"
for request in "u_priya GET /vesting/sh_rahul" "u_priya GET /captable" "u_nobody GET /vesting/sh_priya"; do
  read -r user method path <<<"$request"
  code="$(curl -sS -o /dev/null -w '%{http_code}' -X "$method" "$BASE_URL$path" -H "X-User-Id: $user")"
  echo "  $user $method $path -> $code"
done
echo

echo "=== Last 3 audit records (GET /audit as admin)"
curl -sS "$BASE_URL/audit?limit=3" -H "X-User-Id: u_arjun" |
  jq -r '.[] | "  \(.ts) \(.user_id) [\(.outcome)] \(.latency_ms) ms, model \(.model), as_of \(.as_of)\n    Q: \(.question)\n    tools: \([.tool_calls[].name] | join(", ")) | chunks: \(.chunk_ids | length)\n    A: \(.answer // .error | .[0:160])"'
