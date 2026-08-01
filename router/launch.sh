#!/usr/bin/env bash
# router/launch.sh — start the single-endpoint stack:
#   standalone embedding (:8085) + colbert reranker (:8086) + router (:8080).
# Strips the gpu-guard shim per FACTS.md. Waits for health on each port.
# NOTE: per project constraints this script is an artifact — it is NOT run
# automatically by tests; run it manually to bring the stack up.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=serve.conf
source "$HERE/serve.conf"

LOG_DIR="${LLAMA_LOG_DIR:-/tmp/llama_logs}"
PID_DIR="${LLAMA_PID_DIR:-/tmp/llama_pids}"
mkdir -p "$LOG_DIR" "$PID_DIR"

# --- helpers ---------------------------------------------------------------
port_busy() { ss -tlnH "sport = :$1" 2>/dev/null | grep -q .; }

wait_health() {
  local port="$1" name="$2" tries="${3:-120}"
  local i
  for ((i=0; i<tries; i++)); do
    if curl -sf -m 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[OK] $name healthy on :$port"
      return 0
    fi
    sleep 1
  done
  echo "[ERR] $name on :$port did not become healthy in ${tries}s"
  return 1
}

start_daemon() {
  # start_daemon <name> <port> <args...>
  local name="$1" port="$2"; shift 2
  if port_busy "$port"; then
    echo "[SKIP] $name: port $port already busy"
    return 0
  fi
  echo "[START] $name -> :$port (log: $LOG_DIR/$name.log)"
  # GPU_ENV strips LD_PRELOAD + forces CUDA_VISIBLE_DEVICES=0 (FACTS.md)
  nohup $GPU_ENV "$LLAMA_BIN" "$@" > "$LOG_DIR/$name.log" 2>&1 &
  echo $! > "$PID_DIR/$name.pid"
  disown $! 2>/dev/null || true
}

# --- standalone always-on services (never evicted by the router) -----------
# Paths must match presets.ini / llama-serve-v2.sh registry.
EMBED_MODEL="/home/thomas/models/Embedding-350M-GGUF/LFM2.5-Embedding-350M-F16.gguf"
COLBERT_MODEL="/home/thomas/models/ColBERT-350M-GGUF/LFM2.5-ColBERT-350M-F16.gguf"

start_daemon lfm-embedding "$EMBEDDING_PORT" \
  -m "$EMBED_MODEL" \
  --host 0.0.0.0 --port "$EMBEDDING_PORT" \
  --alias lfm2.5-embedding-350m \
  -ngl 99 -c 8192 -fa on --embedding --pooling mean

start_daemon lfm-colbert "$RERANKER_PORT" \
  -m "$COLBERT_MODEL" \
  --host 0.0.0.0 --port "$RERANKER_PORT" \
  --alias lfm2-colbert-350m \
  -ngl 99 -c 8192 -fa on --rerank --pooling rank

wait_health "$EMBEDDING_PORT" lfm-embedding
wait_health "$RERANKER_PORT" lfm-colbert

# --- router on :8080 --------------------------------------------------------
if port_busy "$LLAMA_ROUTER_PORT"; then
  echo "[SKIP] router: port $LLAMA_ROUTER_PORT already busy"
else
  echo "[START] router -> :$LLAMA_ROUTER_PORT (log: $LOG_DIR/router.log)"
  nohup $GPU_ENV "$LLAMA_BIN" \
    --host 0.0.0.0 --port "$LLAMA_ROUTER_PORT" \
    --models-preset "$PRESETS_FILE" \
    --models-max "$MODELS_MAX" \
    --sleep-idle-seconds "$SLEEP_IDLE" \
    > "$LOG_DIR/router.log" 2>&1 &
  echo $! > "$PID_DIR/router.pid"
  disown $! 2>/dev/null || true
fi

wait_health "$LLAMA_ROUTER_PORT" router 180
echo "[DONE] stack up: router :$LLAMA_ROUTER_PORT, embedding :$EMBEDDING_PORT, reranker :$RERANKER_PORT"
