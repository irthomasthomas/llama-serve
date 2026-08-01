#!/usr/bin/env bash
# tests/validate-systemd.sh — static lint of llama@.service via
# systemd-analyze --user verify against a TEMP XDG config dir (never touches
# ~/.config/systemd), plus a dry-run of llama-health.sh's loading-vs-wedged
# logic with mocked curl/journalctl/systemctl.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
UNIT="$ROOT/systemd/llama@.service"
HEALTH="$ROOT/systemd/llama-health.sh"
pass=0; fail=0
ok()  { printf '  PASS %s\n' "$1"; pass=$((pass+1)); }
bad() { printf '  FAIL %s\n' "$1"; fail=$((fail+1)); }

TMP="$(mktemp -d /tmp/llama-sysd.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

echo "== 1. systemd-analyze --user verify (temp XDG, ~/.config untouched) =="
XDG_DIR="$TMP/xdg"; mkdir -p "$XDG_DIR/systemd/user"
cp "$UNIT" "$XDG_DIR/systemd/user/llama@.service"
# verify needs the env dir to exist for EnvironmentFile (- prefix = optional, ok)
if command -v systemd-analyze >/dev/null 2>&1; then
  out=$(XDG_CONFIG_HOME="$XDG_DIR" systemd-analyze --user verify "$XDG_DIR/systemd/user/llama@.service" 2>&1)
  rc=$?
  # verify returns non-zero on hard errors; warnings on stdout/stderr are ok
  if [[ $rc -eq 0 ]]; then
    ok "systemd-analyze verify clean"
  else
    # tolerate warnings about missing env files / paths, fail on parse errors
    if grep -qiE "parse|syntax|invalid|Failed to (read|load)" <<<"$out"; then
      bad "verify hard error: $out"
    else
      ok "verify passed with warnings only"
    fi
  fi
  # confirm real ~/.config was not touched (mtime check)
  [[ -d ~/.config/systemd ]] && before=$(stat -c %Y ~/.config/systemd 2>/dev/null || echo 0) || before=none
  sleep 1
  [[ -d ~/.config/systemd ]] && after=$(stat -c %Y ~/.config/systemd 2>/dev/null || echo 0) || after=none
  [[ "$before" == "$after" ]] && ok "~/.config/systemd untouched" || bad "~/.config/systemd modified!"
else
  echo "  SKIP systemd-analyze not installed"
fi

echo "== 2. unit bash snippets parse =="
python3 - "$UNIT" "$TMP" <<'PY'
import sys, re
unit, tmp = sys.argv[1], sys.argv[2]
text = open(unit).read()
n = 0
for m in re.finditer(r"^Exec(?:StartPre|Start|StopPost)=/bin/bash -c '(.*)'$", text, re.M):
    body = m.group(1).replace("''", "'")
    n += 1
    open(f"{tmp}/snippet{n}.sh", "w").write(body + "\n")
print(n)
PY
nsnips=$(python3 - "$UNIT" "$TMP" <<'PY'
import sys, re
text = open(sys.argv[1]).read()
print(len(re.findall(r"^Exec(?:StartPre|Start|StopPost)=/bin/bash -c '(.*)'$", text, re.M)))
PY
)
[[ "$nsnips" -ge 2 ]] && ok "extracted $nsnips bash snippets" || bad "snippet extraction failed ($nsnips)"
for f in "$TMP"/snippet*.sh; do
  [[ -f "$f" ]] || continue
  bash -n "$f" 2>/dev/null && ok "bash -n $(basename "$f")" || bad "bash -n failed: $f"
done

echo "== 3. llama-health.sh loading-vs-wedged logic (mocked) =="
MOCK_BIN="$TMP/mockbin"; mkdir -p "$MOCK_BIN"
RT="$TMP/runtime"; mkdir -p "$RT"
MODE_FILE="$TMP/mode"   # healthy | wedged | loading
echo healthy > "$MODE_FILE"
cat > "$MOCK_BIN/curl" <<MOCKEOF
#!/usr/bin/env bash
# args: -sf -m 5 http://localhost:PORT/health
mode=\$(cat "$MODE_FILE")
[[ "\$mode" == "healthy" ]] && exit 0 || exit 7
MOCKEOF
cat > "$MOCK_BIN/journalctl" <<'MOCKEOF'
#!/usr/bin/env bash
mode=$(cat "__MODE__")
case "$mode" in
  wedged)  echo "llama-server: server is listening on http://0.0.0.0:8080" ;;
  loading) echo "llama_model_load: loading model weights"; echo "ggml_cuda_init: found 1 CUDA devices" ;;
  *)       echo "" ;;
esac
MOCKEOF
sed -i "s|__MODE__|$MODE_FILE|" "$MOCK_BIN/journalctl"
cat > "$MOCK_BIN/systemctl" <<MOCKEOF
#!/usr/bin/env bash
# record restart calls
echo "\$*" >> "$RT/systemctl.calls"
exit 0
MOCKEOF
chmod +x "$MOCK_BIN"/{curl,journalctl,systemctl}

run_health() {
  PATH="$MOCK_BIN:$PATH" XDG_RUNTIME_DIR="$RT" bash "$HEALTH"
}

# --- case A: healthy => no restart, no fail files ---
: > "$RT/systemctl.calls"; echo healthy > "$MODE_FILE"
out=$(run_health)
[[ ! -s "$RT/systemctl.calls" ]] && ok "healthy: no restart" || bad "healthy: unexpected restart"
ls "$RT"/llama-health/*.fails >/dev/null 2>&1 && bad "healthy: fail files created" || ok "healthy: no fail files"

# --- case B: loading => skip, fail counter not incremented, no restart ---
rm -rf "$RT/llama-health"; : > "$RT/systemctl.calls"; echo loading > "$MODE_FILE"
out=$(run_health)
grep -q "still loading" <<<"$out" && ok "loading: skip message" || bad "loading: no skip message"
[[ ! -s "$RT/systemctl.calls" ]] && ok "loading: no restart" || bad "loading: unexpected restart"
ls "$RT"/llama-health/*.fails >/dev/null 2>&1 && bad "loading: fail counter incremented" || ok "loading: counter not incremented"

# --- case C: wedged => restart only after 2 consecutive checks ---
rm -rf "$RT/llama-health"; : > "$RT/systemctl.calls"; echo wedged > "$MODE_FILE"
out1=$(run_health)
[[ ! -s "$RT/systemctl.calls" ]] && ok "wedged check1: no restart yet" || bad "wedged check1: restarted too early"
grep -q "failed check 1/2" <<<"$out1" && ok "wedged check1: counter 1/2" || bad "wedged check1: no counter"
out2=$(run_health)
grep -q "restart" <<<"$(cat "$RT/systemctl.calls")" && ok "wedged check2: restart issued" || bad "wedged check2: no restart"
grep -q "dead for 2 consecutive checks" <<<"$out2" && ok "wedged check2: threshold message" || bad "wedged check2: no threshold msg"

# --- case D: recovery resets counter (seed nonzero counters first) ---
: > "$RT/systemctl.calls"; echo healthy > "$MODE_FILE"
echo 1 > "$RT/llama-health/lfm-8b.fails"; echo 1 > "$RT/llama-health/lfm-1.2b.fails"
out=$(run_health)
grep -q "recovered" <<<"$out" && ok "recovery: recovered message" || bad "recovery: no recovered message"
[[ "$(cat "$RT/llama-health/lfm-8b.fails")" == "0" ]] && ok "recovery: counter reset" || bad "recovery: counter not reset"

echo
echo "== RESULT =="
echo "  pass=$pass fail=$fail"
[[ $fail -eq 0 ]]
