#!/usr/bin/env bash
# Dry-run self-test: exact generated command lines via stub llama-server.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
SCRIPT="$ROOT/llama-serve-v2.sh"
pass=0; fail=0
ok()  { printf '  PASS %s\n' "$1"; pass=$((pass+1)); }
bad() { printf '  FAIL %s\n' "$1"; fail=$((fail+1)); }
TMP="$(mktemp -d /tmp/llama-dryrun.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
PID_DIR="$TMP/pids"; mkdir -p "$PID_DIR"
LOG_DIR="$TMP/logs"; mkdir -p "$LOG_DIR"
ARGS_FILE="$TMP/llama.args"
STUB="$TMP/llama-server"
{
  echo '#!/usr/bin/env bash'
  echo "printf '%s\n' \"\$*\" > \"$ARGS_FILE\""
  echo 'echo "stub: LD_PRELOAD=${LD_PRELOAD-<unset>} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"'
  echo 'sleep 30'
} > "$STUB"
chmod +x "$STUB"
echo "== 0. prerequisites =="
[[ -x "$STUB" ]] && ok "stub binary executable" || bad "stub missing"
run_launcher() {
  LLAMA_BIN="$STUB" LLAMA_PID_DIR="$PID_DIR" LLAMA_LOG_DIR="$LOG_DIR" "$SCRIPT" "$@"
}
cleanup() { rm -f "$PID_DIR"/*.pid "$ARGS_FILE" 2>/dev/null; }
echo "== 1. start lfm-8b (speculative draft expected) =="
out=$(run_launcher start lfm-8b 2>&1)
[[ $? -eq 0 ]] && ok "launcher exit 0" || { bad "launcher rc"; echo "$out"; }
grep -q 'speculative decoding enabled' <<<"$out" && ok "draft info line printed" || bad "no draft info line"
args="$(cat "$ARGS_FILE" 2>/dev/null)"
[[ -n "$args" ]] && ok "stub captured argv" || bad "no argv captured"
check() { grep -qF -- "$1" <<<"$args" && ok "lfm-8b arg: $1" || bad "lfm-8b missing: $1"; }
check "-m /home/thomas/models/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q6_K.gguf"
check "--port 8080"
check "-c 65536"
check "-ngl 99"
check "-np 1"
check "-fa on"
check "--cache-reuse 256"
check "--jinja"
check "--model-draft /home/thomas/models/LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
check "--spec-draft-ngl 99"
check "--spec-draft-n-max 8"
check "--spec-draft-n-min 2"
check "--alias lfm2.5-8b-a1b"
log="$LOG_DIR/lfm-8b.log"
grep -q 'LD_PRELOAD=<unset>' "$log" && ok "LD_PRELOAD stripped" || bad "LD_PRELOAD still set"
grep -q 'CUDA_VISIBLE_DEVICES=0' "$log" && ok "CUDA_VISIBLE_DEVICES=0 forced" || bad "CUDA_VISIBLE_DEVICES wrong"
[[ -f "$PID_DIR/lfm-8b.pid" ]] && ok "pid file written" || bad "no pid file"
st_out=$(run_launcher status 2>/dev/null); grep -q "running:" <<<"$st_out" && ok "status shows running" || bad "status not running"
run_launcher stop lfm-8b >/dev/null 2>&1; cleanup
echo "== 2. start lfm-1.2b (NO draft flags expected) =="
out=$(run_launcher start lfm-1.2b 2>&1)
args="$(cat "$ARGS_FILE" 2>/dev/null)"
grep -qF -- "--port 8081" <<<"$args" && ok "lfm-1.2b arg: --port 8081" || bad "lfm-1.2b port wrong"
grep -qF -- "-c 128000" <<<"$args" && ok "lfm-1.2b arg: -c 128000" || bad "lfm-1.2b ctx wrong"
grep -qE -- "--model-draft|--spec-draft" <<<"$args" && bad "lfm-1.2b has spec-draft flags" || ok "lfm-1.2b has no spec-draft flags"
run_launcher stop lfm-1.2b >/dev/null 2>&1; cleanup
echo "== 3. unknown model rejected =="
run_launcher start nope-model 2>&1 | grep -q 'Unknown model' && ok "unknown model rejected" || bad "unknown model not rejected"
echo
echo "== RESULT =="
echo "  pass=$pass fail=$fail"
[[ $fail -eq 0 ]]
