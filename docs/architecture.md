# llama-serve-modern — Architecture

> **Box:** RTX 3060 (12 GiB VRAM, CC 8.6, PCIe 4.0) · 64 GiB RAM · CUDA · llama.cpp b9139

---

## Before → After

```
        BEFORE  (legacy: one llama-server per model, static ports)

  ┌────────┐
  │ Client │──► :8080  lfm-8b          ┐
  │ tgbot  │──► :8081  lfm-1.2b        │
  │ scripts│──► :8085  embedding       ├── N processes · N ports
  │ etc.   │──► :8086  colbert         │   flags drift per model
  │        │──► :8088  vl-450m         │   no eviction control
  └────────┘──► :8082..8090 on-demand  ┘   manual VRAM juggling
                                         no health watchdog
                                         fragile bash restarts


        AFTER  (router + guard + systemd)

  ┌────────┐     ┌──────────┐     ┌─────────────────────────────────┐
  │ Client │────►│  guard   │────►│  ROUTER :8080  (models-max 2)     │
  │        │     │  :8091   │     │   ├── lfm-8b (+1.2b spec draft)  │
  │        │     └──────────┘     │   ├── lfm-1.2b  (load-on-startup)│
  │        │                      │   ├── vibethinker-3b   (swap)    │
  │        │     ┌──────────────┐  │   └── …               (swap)    │
  │        │────►│ standalone   │  └─────────────────────────────────┘
  │        │     │ :8085 embed  │
  │        │     │ :8086 colbert│  ┄┄┄ always-on, outside router budget
  └────────┘     └──────────────┘

  systemd llama@<model>  ·  llama-health.sh timer  ·  env-file config
```

---

## Five Changes

| # | Change | Artifacts |
|---|--------|-----------|
| 1 | llama-serve-v2.sh flag fixes | `env/*.env` (11 files), `legacy/llama-serve-v1.sh` |
| 2 | Speculative decoding w/ 1.2B draft | `env/lfm-8b.env` (`-md …`), `router/presets.ini` `[lfm-8b]` |
| 3 | systemd + health watchdog | `systemd/llama@.service`, `systemd/llama-health.sh`, `systemd/README-install.md`, `env/*.env` |
| 4 | Router single endpoint :8080 | `router/presets.ini`, `router/serve.conf`, `router/README.md` |
| 5 | Eviction guard :8091 | `router/llama-guard.py`, `tests/test_guard.py` |

### 1. llama-serve-v2.sh flag fixes

The legacy launcher (`legacy/llama-serve-v1.sh`) passed ad-hoc flag strings per
model via a bash registry.  Several flags were wrong or missing for b9139.

The v2 **base flags** — verified against `llama-server --help` and documented in
`FACTS.md` — are now baked into every `env/*.env`:

```
-ngl 99  -fa on  --cache-reuse 256  -ctk q8_0  -ctv q8_0  --jinja  -np 1
```

Confirmed b9139 flags: `--cache-reuse`, `--models-preset`, `--models-max`,
`--sleep-idle-seconds`, `--spec-draft-model`/`-md`, `--spec-draft-n-max`,
`--spec-draft-ngl`, `--spec-draft-p-min`, `-fa`, `-ctk`, `-ctv`, `--jinja`,
`--rerank`, `--pooling`, `--embedding`, `--mmproj`.

### 2. Speculative decoding with 1.2B draft

The primary chat model (`lfm-8b`, LFM2.5-8B-A1B Q6\_K) uses the 1.2B instruct
model (Q4\_K\_M) as a speculative draft:

```
-md models/LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q4_K_M.gguf
--spec-draft-ngl 99  --spec-draft-n-max 16
```

The 1.2B model is **dual-purpose**: spec draft for the 8B AND a standalone
router-resident chat model (`load-on-startup = true` in `presets.ini`). No extra
VRAM cost — the draft is already loaded for small-chat requests.

### 3. systemd units + health watchdog

Each model runs as a hardened `systemd --user` unit (`llama@<name>.service`):

- `Type=simple` (b9139 lacks `sd_notify` — verified: no libsystemd linkage)
- `ExecStartPre` port-collision guard (sleep-free; systemd retries on exit 1)
- `LD_PRELOAD=` + `CUDA_VISIBLE_DEVICES=0` in unit Environment (defeats gpu-guard)
- `Restart=on-failure`, `StartLimitBurst=5` / `StartLimitIntervalSec=300`
- All output → journald (no file redirects)
- `ExecStopPost` cleans stale pid files

`llama-health.sh` (systemd timer, every 2 min):
- Checks `/health` on 8080 + 8081
- Restarts only after **2 consecutive failures** AND journal shows prior
  `server is listening` (never restarts a model still loading)

### 4. Router single endpoint :8080

```
llama-server --models-preset router/presets.ini \
             --models-max 2 --sleep-idle-seconds 300 \
             --port 8080 --host 0.0.0.0
```

One OpenAI-compatible endpoint replaces N per-model servers. Clients select
models via the `model` field. The router loads/evicts instances on demand,
keeping at most 2 chat/VL models resident.

Tiny always-on services (embedding :8085, reranker :8086) run as **standalone**
processes outside the router's eviction budget — zero cold-start, never evicted.

### 5. Eviction guard :8091

`router/llama-guard.py` — stdlib-only HTTP proxy in front of the router:

```
client ──► :8091 guard ──► :8080 router
```

| Condition | Action | HTTP |
|-----------|--------|------|
| Model is resident | `allow` | forward → 200 |
| Non-resident + `X-Allow-Swap: 1` | `override` | forward → 200 |
| Non-resident, no override | `block` | **409** `model_swap_guard` |
| `--dry-run` | `dry-run` | synthetic 200, backend untouched |

The 409 body tells the client exactly what happened and what to do:

```json
{"error":{
  "type":"model_swap_guard",
  "message":"vibethinker-3b requires evicting a resident model (~3-5s reload)",
  "alternatives":["lfm2.5-8b-a1b","lfm2.5-1.2b-instruct"],
  "override":"resend with header X-Allow-Swap: 1 or choose an alternative model"
}}
```

---

## Trade-offs: 12 GiB VRAM vs 64 GiB RAM

The fundamental constraint is **VRAM, not RAM**. With ~11.5 GiB available for
model weights (after ~300 MiB desktop compositing), only a fixed resident set
fits. Everything else must load on demand.

### mmap → page-cache reload (~3-5 s)

llama-server uses `mmap()` to map GGUF weight files. When a model is evicted
from VRAM its pages stay hot in the Linux page cache (system RAM). On reload:

| Step | Time |
|------|------|
| Disk I/O | **0 s** — pages already in 64 GiB page cache |
| PCIe 4.0 transfer (6.5 GiB → VRAM at ~16 GB/s) | ~0.5 s |
| CUDA context + KV cache allocation | ~2-4 s |
| **Total cold-reload from page cache** | **~3-5 s** |

### Resident vs cold-reload TTFT

| State | TTFT (first token) | Relative |
|-------|---------------------|----------|
| Resident (weights hot in VRAM) | ~50-100 ms | 1× |
| Cold reload from page cache | ~3-5 s | ~50-100× |
| First-ever load from NVMe | ~10-15 s | ~1000× |

The guard exists to **prevent accidental swaps** that would stall the Telegram
bot for seconds. The 409 + alternatives lets the client retry with a resident
model instantly rather than waiting for an eviction cycle.

### Why not keep all models in VRAM?

12 GiB VRAM ÷ ~2 GiB average per instance ≈ 5-6 models max — and the 8B model
alone needs 7.5 GiB (weights + KV + compute). There is no headroom for a third
chat model without evicting one. `--models-max 2` + standalone services fill the
card exactly.

---

## Resident set VRAM budget

| Model | Role | VRAM (GiB) |
|-------|------|------------|
| `lfm-8b` (Q6\_K + 1.2B spec draft) | Primary chat | 7.5 |
| `lfm-1.2b` (Q4\_K\_M) | Spec draft + small chat | 0.8 |
| `lfm-embedding` (F16, 350M) | Embeddings (:8085 standalone) | 0.7 |
| `lfm-colbert` (F16, 350M) | Reranker (:8086 standalone) | 0.7 |
| `lfm2-vl-450m` (Q8\_0) | Vision-language (router load-on-startup) | 0.6 |
| **Total** | | **~10.3 GiB** |

Remaining **~1.2 GiB**: KV cache growth headroom, compute buffers, CUDA context,
desktop compositing. Tight but feasible because `-ctk q8_0 -ctv q8_0` halves KV
footprint and `--models-max 2` caps concurrent instances.

**Swap pool** (on-demand, evict the 8B): `vibethinker-3b`, `fastcontext-rl`,
`fastcontext-sft`, `lfm-8b-base`, `lfm2-vl-1.6b`.

---

## Cutover plan

> Do NOT start/stop real models during artifact development.
> This plan is for the production cutover only.

1. **Verify artifacts green** — `python3 tests/test_guard.py -v` and
   `bash tests/test_presets.sh`.
2. **Install units + env files:**
   ```bash
   cp systemd/llama@.service ~/.config/systemd/user/
   cp env/*.env ~/ai/models/
   cp systemd/llama-health.sh ~/ai/
   systemctl --user daemon-reload
   ```
3. **Stop legacy servers:**
   ```bash
   ~/ai/llama-serve.sh stop all
   ```
4. **Start modern stack:**
   ```bash
   systemctl --user start llama@lfm-8b
   systemctl --user start llama@lfm-embedding
   systemctl --user start llama@lfm-colbert
   systemctl --user enable --now llama-health.timer
   python3 ~/work/llama-serve-modern/router/llama-guard.py \
       --listen-port 8091 --backend-host 127.0.0.1 --backend-port 8080 &
   ```
5. **Verify endpoints:**
   ```bash
   curl localhost:8080/v1/models      # router
   curl localhost:8091/v1/models      # guard proxy
   curl localhost:8085/v1/embeddings  # standalone embedding
   ```
6. **Update clients** to use `model` field selection on :8080 (or :8091 guarded).

---

## Rollback

If the modern stack misbehaves, revert to the legacy one-server-per-model
launcher:

```bash
systemctl --user stop 'llama@*'
kill $(pgrep -f llama-guard.py)

# Legacy: restores static :8080 (lfm-8b) + :8081 (lfm-1.2b)
bash ~/ai/llama-serve.sh start lfm-8b lfm-1.2b
```

---

## Gotchas

### LD\_PRELOAD gpu-guard shim → silent CPU-only

Agent shells export:

```
LD_PRELOAD=/home/thomas/ai/gpu-guard/vk_layer_cpu_only.so
CUDA_VISIBLE_DEVICES=""
```

This forces llama-server onto CPU **silently** — no error, just ~50× slower
inference. **Every launch path must strip it:**

| Path | Mitigation |
|------|------------|
| `systemd/llama@.service` | `Environment=LD_PRELOAD=` + `CUDA_VISIBLE_DEVICES=0` |
| Router / standalone launch | `env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0 llama-server …` |
| `router/llama-guard.py` | N/A — pure HTTP, no GPU access |

Diagnose:
```bash
cat /proc/$(pgrep -f llama-server)/environ | tr '\0' '\n' | grep LD_PRELOAD
```
If non-empty, the shim is active and the model is running on CPU.
