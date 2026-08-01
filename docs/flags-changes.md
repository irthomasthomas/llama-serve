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

## Self-speculative decoding via ngram-mod (2026-08-01, recommended)

Draft-MODEL speculation is impossible for lfm-8b: HF check of every smaller
LFM2.5 (1.2B / 350M / 230M) shows all use vocab 65536, but the 8B uses 128000
(`model_type: lfm2_moe`). b9139 refuses mismatched draft/target vocabs.

b9139 instead supports SELF-speculative ngram (no second model, no vocab
constraint, ~0 extra VRAM):

    --spec-type ngram-mod           # self-tuning lookup (best)
    # also: ngram-simple, ngram-map-k, ngram-map-k4v, ngram-cache, draft-eagle3

A/B benchmark, RTX 3060, lfm-8b Q6_K, ctx 65536, -np 1 -fa on, steady state,
3 workload types (baseline == same flags, no --spec-type):

| workload             | baseline | ngram-simple | ngram-mod | mod delta |
|----------------------|----------|--------------|-----------|-----------|
| repetitive (count)   | 161.9    | 155.5 (-4%)  | 984.6     | +508%     |
| code gen             | 162.0    | 152.1 (-6%)  | 213.8     | +32%      |
| prose                | 161.8    | 154.0 (-5%)  | 171.9     | +6%       |

ngram-mod acceptance ~48-49%. **ngram-simple is a net LOSS (-4 to -6%)** at
steady state (acceptance only ~17%; the "45%" first reported was a warm-up
burst) — do NOT use it. ngram-mod self-tunes its lookup and wins everywhere,
massively on repetitive/self-referential output (lists, code, RAG). Enabled
as `--spec-type ngram-mod` on lfm-8b in the launcher registry and
router/presets.ini. Output correctness verified identical (speculation is
lossless: drafted tokens are re-verified by the target model).
