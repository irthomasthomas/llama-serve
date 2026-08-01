# llama-serve

Modernized multi-model launcher for llama.cpp `llama-server` (b9139, CUDA) on a
single RTX 3060 12 GB box. Two operating modes share one model registry:

| Mode | Entry point | Use |
|------|-------------|-----|
| **Per-model** | `llama-serve-v2.sh` | one server per port (legacy-compatible) |
| **Router** | `router/launch.sh` + `router/presets.ini` | single OpenAI endpoint on `:8080`, on-demand model loading |

## Quick start (per-model)

```sh
./llama-serve-v2.sh list            # registry
./llama-serve-v2.sh start lfm-8b    # chat on :8080 (speculative draft on)
./llama-serve-v2.sh start lfm-1.2b  # fast chat on :8081
./llama-serve-v2.sh status          # table + VRAM
./llama-serve-v2.sh vram            # per-process GPU holders
./llama-serve-v2.sh evict --free 6000   # free VRAM, largest ctx first
```

## Quick start (router)

```sh
bash router/launch.sh     # embedding :8085 + reranker :8086 + router :8080
# then: POST :8080/v1/chat/completions {"model":"lfm2.5-8b-a1b", ...}
```

`--models-max 2` keeps at most two router models resident; idle models sleep
after 300 s. Embedding/reranker run **outside** the router so they are never
evicted. `router/llama-guard.py` (optional proxy on `:8091`) answers 409 when a
request would evict a resident model — override with header `X-Allow-Swap: 1`.

## Key changes vs v1

- `-np 1 -fa on -ctk q8_0 -ctv q8_0 --jinja --cache-reuse 256` on every server
- speculative decoding: `lfm-8b` drafts off `lfm-1.2b` (`--model-draft` + spec flags)
- **gpu-guard shim fix** (FACTS.md): all launch paths run
  `env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0 …` — without it llama-server
  silently starts CPU-only from agent shells
- multi-tier VRAM probing that works when NVML breaks (sysfs → /proc → logs)

## Layout

```
llama-serve-v2.sh    per-model launcher (registry, start/stop/vram/evict)
legacy/              v1 script for reference
router/              presets.ini, serve.conf, launch.sh, llama-guard.py
systemd/             llama@.service template + llama-health.sh watchdog
env/                 one .env per registry model (systemd EnvironmentFile)
docs/                architecture.md, flags-changes.md
tests/               see below
FACTS.md             verified environment facts (read first)
```

## Tests (artifacts only — no real models started)

| Test | What it proves |
|------|----------------|
| `tests/test_dryrun.sh` | exact generated command lines via stub binary (25 checks) |
| `tests/test_presets.sh` | presets.ini parses, paths exist, flags valid vs `--help` |
| `tests/test_guard.py` | guard proxy logic, in-process (13 cases) |
| `tests/test_guard_subprocess.py` | guard CLI/dry-run/SIGTERM, subprocess level |
| `tests/validate-systemd.sh` | `systemd-analyze verify` in temp XDG + mocked watchdog logic |
| `tests/test_router_live.sh` | **manual** live smoke test against a running stack |

## systemd

See `systemd/README-install.md`. Units are **not** installed by this repo;
copy the template + env files and `systemctl --user enable llama@lfm-8b` etc.

## Links

- Repo: https://github.com/irthomasthomas/llama-serve
- llama.cpp b9139 router docs: `--models-preset`, `--models-max`, `--sleep-idle-seconds`
