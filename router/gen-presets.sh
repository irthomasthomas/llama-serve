#!/usr/bin/env bash
# router/gen-presets.sh — regenerate presets.ini from the llama-serve-v2.sh
# registry so the router and per-model template paths stay in sync.
# Usage: bash router/gen-presets.sh > router/presets.ini
#        bash router/gen-presets.sh --check   (exit 1 if presets.ini is stale)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
REGISTRY_SCRIPT="$ROOT/llama-serve-v2.sh"

# --- extract the registry from llama-serve-v2.sh without executing it ------
# The registry is the declare -A REGISTRY=( ... ) block; source only that.
MODELS_DIR="/home/thomas/models"
export MODELS_DIR
REG_BLOCK=$(awk '/^declare -A REGISTRY=\(/,/^\)/' "$REGISTRY_SCRIPT")
[[ -n "$REG_BLOCK" ]] || { echo "ERR: registry block not found" >&2; exit 1; }
declare -A REGISTRY=()
eval "$REG_BLOCK"

# --- emit -------------------------------------------------------------------
emit() {
  cat <<'HDR'
# ============================================================================
# presets.ini — llama-server router mode (--models-preset), b9139
# GENERATED from llama-serve-v2.sh registry by router/gen-presets.sh.
# Do not edit by hand; edit the registry and re-run the generator.
# ============================================================================
[*]
# Defaults every preset inherits.
ngl          = 99
threads      = 8
threads-batch = 16
cont-batching = true
parallel     = 1
flash-attn   = on
cache-type-k = q8_0
cache-type-v = q8_0
jinja        = true
cache-reuse  = 256

HDR

  # stable order: sort by port
  local name
  for name in $(for n in "${!REGISTRY[@]}"; do
      echo "$(echo "${REGISTRY[$n]}" | cut -d'|' -f2) $n"
    done | sort -n | awk '{print $2}'); do
    local entry file port ctx ngl extra draft
    entry="${REGISTRY[$name]}"
    file=$(echo "$entry" | cut -d'|' -f1)
    port=$(echo "$entry" | cut -d'|' -f2)
    ctx=$(echo "$entry"  | cut -d'|' -f3)
    extra=$(echo "$entry" | cut -d'|' -f5)
    draft=$(echo "$entry" | cut -d'|' -f6); draft="${draft#draft=}"

    echo "[$name]"
    echo "model    = $file"
    echo "ctx-size = $ctx"

    # translate v2 extra-args into INI keys (alias, embedding, rerank,
    # pooling, mmproj, sampling, reasoning) — long --key value pairs
    local k v rest="$extra"
    while [[ -n "$rest" ]]; do
      # pull leading --flag [value]
      if [[ "$rest" =~ ^--([a-z-]+)[[:space:]]+([^[:space:]-][^[:space:]]*)[[:space:]]*(.*)$ ]]; then
        k="${BASH_REMATCH[1]}"; v="${BASH_REMATCH[2]}"; rest="${BASH_REMATCH[3]}"
        echo "$k = $v"
      elif [[ "$rest" =~ ^--([a-z-]+)[[:space:]]*(.*)$ ]]; then
        k="${BASH_REMATCH[1]}"; rest="${BASH_REMATCH[2]}"
        echo "$k = true"
      else
        break
      fi
    done

    # speculative draft wiring
    if [[ -n "$draft" ]]; then
      echo "spec-draft-model = $draft"
      echo "spec-draft-ngl   = 99"
      echo "spec-draft-n-max = 8"
      echo "spec-draft-n-min = 2"
    fi

    # resident set loads on startup (matches serve.conf RESIDENT)
    case "$name" in
      lfm-embedding|lfm-colbert|lfm2-vl-450m|lfm-1.2b)
        echo "load-on-startup = true" ;;
    esac
    echo
  done
}

case "${1:-}" in
  --check)
    # semantic check: every registry model exists as a section with the same
    # model path and ctx-size in presets.ini (formatting/comments may differ)
    python3 - "$HERE/presets.ini" <<'PYEOF2'
import configparser, re, sys
presets_path = sys.argv[1]
script = open(__file__.replace("gen-presets.sh", "../llama-serve-v2.sh")).read()     if False else open(sys.argv[1].replace("router/presets.ini", "llama-serve-v2.sh")).read()
m = re.search(r"declare -A REGISTRY=\((.*?)\n\)", script, re.S)
reg = {}
for line in m.group(1).splitlines():
    mm = re.match(r'\s*\[([^\]]+)\]="([^"]+)"', line)
    if mm:
        f = mm.group(2).split("|")
        reg[mm.group(1)] = (f[0].replace("$MODELS_DIR", "/home/thomas/models"), f[2])
cp = configparser.ConfigParser()
cp.read(presets_path)
bad = 0
for name, (path, ctx) in reg.items():
    if name not in cp:
        print(f"MISSING section [{name}]", file=sys.stderr); bad = 1; continue
    if cp[name].get("model", "") != path:
        print(f"[{name}] model mismatch: {cp[name].get('model')} != {path}", file=sys.stderr); bad = 1
    preset_ctx = cp[name].get("ctx-size", "").split("#")[0].strip()
    # documented router-vs-launcher ctx divergences (VRAM budget on 12GB card)
    CTX_WHITELIST = {"lfm-1.2b": {"32768", "128000"}}
    allowed = CTX_WHITELIST.get(name, {ctx})
    if preset_ctx != ctx and preset_ctx not in allowed:
        print(f"[{name}] ctx-size mismatch: {preset_ctx} != {ctx}", file=sys.stderr); bad = 1
sys.exit(bad)
PYEOF2
    ;;
  *) emit ;;
esac
