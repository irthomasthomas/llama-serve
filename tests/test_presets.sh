#!/usr/bin/env bash
# tests/test_presets.sh — validate router/presets.ini against llama-server.
# Does NOT launch llama-server. Exit non-zero on any failure.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRESETS="$ROOT/router/presets.ini"
LEGACY="$ROOT/legacy/llama-serve-v1.sh"
LLAMA_BIN="${LLAMA_BIN:-/home/thomas/llama.cpp/build_optimized/bin/llama-server}"
HELP_TXT="$(mktemp)"
trap 'rm -f "$HELP_TXT"' EXIT

pass=0; fail=0
ok()   { printf '  \033[32mPASS\033[0m %s\n' "$1"; pass=$((pass+1)); }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=$((fail+1)); }
sec()  { printf '\n== %s ==\n' "$1"; }

# Cache --help once.
"$LLAMA_BIN" --help >"$HELP_TXT" 2>&1 || { echo "cannot run $LLAMA_BIN --help"; exit 2; }

sec "1. INI parses with python configparser"
python3 - "$PRESETS" <<'PY'
import configparser, sys
p = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',';'), strict=True)
p.read(sys.argv[1], encoding="utf-8")
# configparser folds the leading 'version = 1' into a DEFAULT-ish sentinel only
# if it is outside a section; configparser actually rejects bare keys. Accept
# either: a top-level "version" is the only allowed bare line per upstream.
print("  sections:", len(p.sections()), p.sections())
sys.exit(0)
PY
if [[ $? -eq 0 ]]; then ok "configparser parsed presets.ini"; else bad "configparser rejected presets.ini"; fi

sec "2. Preset count == legacy registry count"
preset_n=$(python3 - "$PRESETS" <<'PY'
import configparser, sys
p = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',';'))
p.read(sys.argv[1], encoding="utf-8")
print(sum(1 for s in p.sections() if s != "*"))
PY
)
# Count uncommented REGISTRY entries:  <spaces>[name]="...   (name may contain . - _)
reg_n=$(grep -oE '^[[:space:]]*\[[A-Za-z0-9._-]+\][[:space:]]*=' "$LEGACY" \
        | grep -vE '^[[:space:]]*#' | wc -l)
echo "  presets=$preset_n  registry=$reg_n"
if (( preset_n == reg_n )) && (( preset_n > 0 )); then ok "counts match ($preset_n)"; else bad "count mismatch"; fi

sec "3. Every model/mmproj/draft path exists"
python3 - "$PRESETS" <<'PY'
import configparser, os, sys
p = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',';'))
p.read(sys.argv[1], encoding="utf-8")
path_keys = ("model","mmproj","model-draft","spec-draft-model")
missing=[]
for s in p.sections():
    if s == "*": continue
    for k in path_keys:
        if k in p[s]:
            v = p[s][k].strip()
            if not os.path.exists(v):
                missing.append(f"[{s}] {k} = {v}")
            else:
                print(f"  exists [{s}] {k}")
if missing:
    print("MISSING:\n  " + "\n  ".join(missing), file=sys.stderr)
    sys.exit(1)
PY
if [[ $? -eq 0 ]]; then ok "all referenced files exist"; else bad "some paths missing"; fi

sec "4. Every INI flag key appears in llama-server --help"
python3 - "$PRESETS" "$HELP_TXT" <<'PY'
import configparser, re, sys
presets, help_path = sys.argv[1], sys.argv[2]
p = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',';'))
p.read(presets, encoding="utf-8")
help_txt = open(help_path, encoding="utf-8", errors="replace").read()

# preset-only / structural keys that are intentionally not CLI flags.
allow = {"load-on-startup","stop-timeout"}

# Build the set of flag names --help advertises (long forms w/o leading dashes,
# and short forms). Lines look like:  "-c,   --ctx-size N  ..." or "--top-k N".
flag_names=set()
for line in help_txt.splitlines():
    for raw in line.split():
        tok = raw.rstrip(",.;:()")        # drop punctuation from alias lists, e.g. "--embedding,"
        if tok.startswith("--") and re.match(r'^--[A-Za-z0-9][A-Za-z0-9_-]*$', tok):
            flag_names.add(tok[2:])
        elif re.match(r'^-[A-Za-z0-9]$', tok):
            flag_names.add(tok[1:])
# A few shorthand aliases the parser accepts that --help groups under another.
flag_names |= {"fa","ctk","ctv","cb","np","ngl","mm","md","c","t","tb","ngld"}

bad=[]
for s in p.sections():
    for k,_ in p.items(s):
        if k in allow: continue
        if k not in flag_names:
            bad.append(f"[{s}] {k}")
if bad:
    print("UNKNOWN KEYS:\n  " + "\n  ".join(bad), file=sys.stderr)
    sys.exit(1)
print("  all", sum(len(p.items(s)) for s in p.sections()), "keys validated")
PY
if [[ $? -eq 0 ]]; then ok "all flag keys recognised by --help"; else bad "unknown flag keys"; fi

sec "5. Resident-set VRAM estimate < 11000 MiB"
# Resident set from serve.conf (machine-wide: standalone + router load-on-startup).
# shellcheck disable=SC1091
eval "$(grep -E '^RESIDENT=' "$ROOT/router/serve.conf")"
echo "  resident: ${RESIDENT[*]}"

python3 - "$PRESETS" "${RESIDENT[@]}" <<'PY'
import configparser, os, sys, math
presets = sys.argv[1]
resident = sys.argv[2:]
p = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',';'))
p.read(presets, encoding="utf-8")

PATH_KEYS=("model","mmproj","spec-draft-model")
def weights_mib(sec):
    tot=0
    for k in PATH_KEYS:
        if k in p[sec]:
            try: tot += os.path.getsize(p[sec][k].strip())
            except OSError: pass
    return math.ceil(tot/1048576)

print(f"  {'model':<16}{'ctx':>8}{'weights':>10}{'overhead':>10}{'total':>9}")
total=0
for name in resident:
    sec=name
    if sec not in p.sections():
        print(f"  WARNING: resident '{sec}' has no preset section", file=sys.stderr); continue
    ctx=int(p[sec].get("ctx-size","8192"))
    w=weights_mib(sec)
    # Bounded KV(q8_0)+compute overhead heuristic: min(ctx/128,512) + 256 MiB.
    oh=min(ctx//128,512)+256
    tot=w+oh
    total+=tot
    print(f"  {sec:<16}{ctx:>8}{w:>10}{oh:>10}{tot:>9}")
print(f"  {'SUM':<16}{'':>8}{'':>10}{'':>10}{total:>9}  (limit 11000)")
sys.exit(0 if total < 11000 else 1)
PY
if [[ $? -eq 0 ]]; then ok "resident VRAM estimate < 11000 MiB"; else bad "resident VRAM estimate >= 11000 MiB"; fi

sec "RESULT"
echo "  pass=$pass fail=$fail"
[[ $fail -eq 0 ]]
