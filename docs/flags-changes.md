# Flag Changes: `legacy/llama-serve-v1.sh` -> `llama-serve-v2.sh`

All flag names below were verified against
`/home/thomas/llama.cpp/build_optimized/bin/llama-server --help`
(`--cache-reuse`, `--model-draft`, `--spec-draft-ngl`, `--spec-draft-n-max`,
`--spec-draft-n-min` all present).

| Old flag / wrapper | New flag / wrapper | Reason |
|---|---|---|
| `-np 2` | `-np 1` | Local traffic is single-user; one parallel decode slot halves KV/scratch buffer cost under the 12 GB VRAM budget |
| `-fa auto` | `-fa on` | Force Flash Attention -- LFM2.5 attention is FA-safe on Ampere (CC 8.6) and `auto` can fall back to slow math |
| *(none)* | `--cache-reuse 256` | KV-shift reuse for >=256-token prefix matches, avoiding full re-encode of repeated system prompts |
| *(none)* | `--model-draft <p> --spec-draft-ngl 99 --spec-draft-n-max 8 --spec-draft-n-min 2` | Speculative decoding with the 1.2 B draft model (registry field `draft=`) for ~1.5-2x token throughput |
| `nohup "$LLAMA_BIN"` | `nohup env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0 "$LLAMA_BIN"` | Strips the gpu-guard shim + hidden-GPU var that agent shells export (see note below) |

## gpu-guard shim bug (env wrapper)

Agent/loop shells export two environment variables that silently break CUDA:

* `LD_PRELOAD=/home/thomas/ai/gpu-guard/vk_layer_cpu_only.so`
* `CUDA_VISIBLE_DEVICES=""`

`LD_PRELOAD` injects a CPU-only Vulkan layer shim and
`CUDA_VISIBLE_DEVICES=""` hides every GPU, so `llama-server` starts in
**CPU-only mode with no error** (~100x slower inference).  The
`env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0` wrapper placed between `nohup`
and the binary removes both so the RTX 3060 is actually used.
See `FACTS.md` -> "CRITICAL ENV BUG".

## X-Allow-Swap note

Hot-swap / swap-eligibility of resident models is **not** handled by this
launcher.  The `X-Allow-Swap` gate and its enforcement live in
[`router/llama-guard.py`](../router/llama-guard.py), which decides whether
a request may trigger eviction from the swap pool (see `FACTS.md` ->
"swap pool").  `llama-serve-v2.sh` only starts/stops individual servers;
it does not emit or honour `X-Allow-Swap` itself.

---

## Live-testing corrections (2026-08-01)

The speculative-decoding flag set above was **validated against `--help` but
NOT against a real load**. First live run of lfm-8b + lfm-1.2b draft showed:

```
common_speculative_are_compatible: draft model bos tokens must match target
common_speculative_impl_draft_simple: the target and draft vocabs are not compatible
srv  load_model: failed to initialize speculative decoding context
```

**lfm-1.2b (vocab 65536) is vocab-INCOMPATIBLE with lfm-8b (vocab 128000)** —
llama-server b9139 silently disables speculation and serves without it. The
draft wiring was removed from both the launcher registry and presets.ini. The
`draft=` registry field remains for any future SAME-vocab draft pair.

Other live findings on the RTX 3060 12 GB:
- lfm-1.2b router ctx cut 128000 -> 32768 (1.5 GB q8 KV was squeezing lfm-8b
  off the card; the launcher keeps 128000, divergence whitelisted in
  gen-presets.sh --check).
- lfm2-vl-450m lost load-on-startup: with --models-max 2, one free slot lets
  a chat swap evict only ONE resident model (lfm-1.2b), avoiding OOM wedges.
- vibethinker-3b (ctx 131072) cannot co-reside with lfm-8b (12 GB card);
  it remains a swap-pool model.
- lfm-8b is a REASONING model: short max_tokens yields empty `content` with
  output in `reasoning_content` (finish_reason=length). Verified 154 tok/s.
