#!/usr/bin/env bash
# tests/test_router_live.sh — MANUAL live smoke test (NOT run by CI).
# Requires the stack to be up: bash router/launch.sh
# Verifies: router /v1/models on :8080, a chat completion with
# model=lfm2.5-8b-a1b, and the standalone embedding (:8085) + rerank (:8086)
# endpoints.
set -uo pipefail
pass=0; fail=0
ok()  { printf '  PASS %s\n' "$1"; pass=$((pass+1)); }
bad() { printf '  FAIL %s\n' "$1"; fail=$((fail+1)); }

ROUTER="${ROUTER_URL:-http://127.0.0.1:8080}"
EMBED="${EMBED_URL:-http://127.0.0.1:8085}"
RERANK="${RERANK_URL:-http://127.0.0.1:8086}"

echo "== 0. router reachable =="
if curl -sf -m 5 "$ROUTER/health" >/dev/null; then
  ok "router /health on :8080"
else
  bad "router not reachable — run: bash router/launch.sh"
  echo "  pass=$pass fail=$fail"; exit 1
fi

echo "== 1. /v1/models lists presets =="
models_json=$(curl -sf -m 10 "$ROUTER/v1/models")
grep -q 'lfm2.5-8b-a1b' <<<"$models_json" && ok "lfm2.5-8b-a1b listed" || bad "lfm2.5-8b-a1b missing"
grep -q 'lfm2.5-1.2b-instruct' <<<"$models_json" && ok "lfm2.5-1.2b-instruct listed" || bad "lfm2.5-1.2b-instruct missing"

echo "== 2. chat completion (model=lfm2.5-8b-a1b) =="
resp=$(curl -sf -m 120 "$ROUTER/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"lfm2.5-8b-a1b","messages":[{"role":"user","content":"Reply with exactly: pong"}],"max_tokens":16,"temperature":0}')
grep -q '"content"' <<<"$resp" && ok "chat completion returned content" || bad "no content in response"
grep -qi 'pong' <<<"$resp" && ok "model replied 'pong'" || bad "unexpected reply: $(head -c200 "$resp" 2>/dev/null || echo "$resp" | head -c200)"

echo "== 3. standalone embedding (:8085) =="
emb=$(curl -sf -m 30 "$EMBED/v1/embeddings" \
  -H 'Content-Type: application/json' \
  -d '{"model":"lfm2.5-embedding-350m","input":"hello world"}')
grep -q '"embedding"' <<<"$emb" && ok "embedding vector returned" || bad "no embedding: $(echo "$emb" | head -c200)"

echo "== 4. standalone reranker (:8086) =="
rr=$(curl -sf -m 30 "$RERANK/v1/rerank" \
  -H 'Content-Type: application/json' \
  -d '{"model":"lfm2-colbert-350m","query":"capital of france","documents":["Paris is the capital of France.","The sky is blue."]}')
grep -qE '"relevance_score"|"score"' <<<"$rr" && ok "rerank scores returned" || bad "no rerank scores: $(echo "$rr" | head -c200)"

echo
echo "== RESULT =="
echo "  pass=$pass fail=$fail"
[[ $fail -eq 0 ]]
