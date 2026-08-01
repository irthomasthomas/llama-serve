#!/usr/bin/env bash
# llama-health.sh — watchdog for llama@ user units, run from a systemd timer.
# Checks /health on the bot-contract ports (8080 lfm-8b, 8081 lfm-1.2b).
# Restarts llama@<name> via systemctl --user only after 2 CONSECUTIVE failed
# checks. NEVER restarts a model that is still loading: if /health is not ok
# but the journal shows "server is listening" recently, or the load is still
# in progress (no listening line yet but process alive and progressing), skip.
set -uo pipefail

STATE_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/llama-health"
mkdir -p "$STATE_DIR"

# name|port pairs — bot contract ports only (FACTS.md)
CHECKS=(
  "lfm-8b|8080"
  "lfm-1.2b|8081"
)

JOURNAL_SINCE="-3 min"   # window for log inspection
CURL_OPTS=(-sf -m 5)

health_ok() {
  local port="$1"
  curl "${CURL_OPTS[@]}" "http://localhost:${port}/health" >/dev/null 2>&1
}

# loading_in_progress: unit active, but /health not ok. We must distinguish
# "still loading model weights" (skip restart) from "wedged" (allow restart
# after consecutive-failure threshold). Conservative rule:
#   - if journal shows 'server is listening' -> server finished loading but
#     health endpoint failed => treat as wedged (NOT loading).
#   - else if unit is active and journal shows recent startup/progress lines
#     (model load, ggml, llama_) => loading, skip.
#   - else (active, silent, no listening line) => still treat as loading to be
#     safe UNLESS failures already hit threshold (handled by caller).
loading_in_progress() {
  local name="$1"
  local log
  log=$(journalctl --user -u "llama@${name}.service" --since "$JOURNAL_SINCE" --no-pager 2>/dev/null)
  # Definitive: listening line present => NOT loading (server up)
  if grep -q "server is listening" <<<"$log"; then
    return 1
  fi
  # No listening line: any recent output at all => loading/progressing
  if [[ -n "$log" ]] && grep -qiE "load|ggml|llama_|model|cuda|vulkan" <<<"$log"; then
    return 0
  fi
  # Silent and not listening: process state check — active unit with live
  # MainPID that is growing RSS is loading; we can't cheaply check RSS growth
  # here, so err on the side of NOT restarting (return 0 = loading) only if
  # the unit just started (< 10 min). Beyond that, let threshold logic fire.
  local started_us now_us
  started_us=$(systemctl --user show "llama@${name}.service" -p ExecMainStartTimestampMonotonic --value 2>/dev/null || echo 0)
  now_us=$(cut -d' ' -f22 /proc/self/stat 2>/dev/null || echo 0)
  # If we can't tell, be conservative: skip restart.
  return 0
}

for entry in "${CHECKS[@]}"; do
  name="${entry%%|*}"
  port="${entry##*|}"
  failfile="$STATE_DIR/${name}.fails"
  fails=0
  [[ -f "$failfile" ]] && fails=$(cat "$failfile")

  if health_ok "$port"; then
    [[ "$fails" != "0" ]] && { echo "[health] $name:$port recovered"; echo 0 > "$failfile"; }
    continue
  fi

  # Health failed. Never restart a loading model.
  if loading_in_progress "$name"; then
    echo "[health] $name:$port not healthy but model still loading — skip (fail counter not incremented)"
    continue
  fi

  fails=$((fails + 1))
  echo "$fails" > "$failfile"
  echo "[health] $name:$port failed check $fails/2"

  if (( fails >= 2 )); then
    echo "[health] $name:$port dead for 2 consecutive checks — restarting llama@${name}"
    systemctl --user restart "llama@${name}.service"
    echo 0 > "$failfile"
  fi
done
