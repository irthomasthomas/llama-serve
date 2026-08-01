# Environment facts (verified 2026-07-30)
- llama-server: /home/thomas/llama.cpp/build_optimized/bin/llama-server (b9139, CUDA enabled)
- CONFIRMED flags: --cache-reuse N; --models-preset INI; --models-max N; --sleep-idle-seconds;
  --spec-draft-model/-md; --spec-draft-n-max (def 16); --spec-draft-n-min; --spec-draft-ngl/-ngld;
  --spec-draft-p-min; -fa on/off/auto; -np; -ctk/-ctv; --jinja; --rerank; --pooling; --embedding; --mmproj
- ROUTER: llama-server --models-preset presets.ini --models-max 2 --port 8080 (OpenAI-compatible, model field selects)
- CRITICAL ENV BUG: agent shells export LD_PRELOAD=/home/thomas/ai/gpu-guard/vk_layer_cpu_only.so and
  CUDA_VISIBLE_DEVICES="" -> llama-server starts CPU-only silently. ALL launch paths must strip them:
  env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0 ...
- VRAM: 12288 MiB total, RTX 3060 (Ampere, CC 8.6, PCIe 4.0). Desktop uses ~300 MiB.
- Weights: 8b Q6_K=6.5G; 1.2b Q4_K_M=0.68G; embed/colbert F16=0.35G each; vl-450m Q8_0=0.5G
- Telegram bot contract: ports 8080 (default chat lfm2.5-8b-a1b) and 8081 (lfm2.5-1.2b) MUST respond;
  external services: 8083=fastcontext, 8085=embedding, 8086=reranker, 8088=vl-450m
- Existing: systemd user template ~/.config/systemd/user/llama@.service -> ~/ai/llama_service.sh %i;
  env files ~/ai/models/<name>.env (keys: MODEL_FILE, PORT, ARGS)
- Registry (name|file|port|ctx|ngl|extra) in legacy/llama-serve-v1.sh (copy of production ~/ai/llama-serve.sh)
- Resident set target ~10.3 GiB; swap pool: vibethinker-3b, fastcontext-rl, fastcontext-sft, lfm-8b-base,
  lfm2-vl-1.6b. Swap evicts only lfm-8b (returns ~3-5s via page cache)
- Repo root: /home/thomas/work/llama-serve-modern; push: git@github.com:irthomasthomas/llama-serve.git
  (GITHUB_TOKEN env var belongs to wrong account; run `unset GITHUB_TOKEN` before gh)
- Do NOT start/stop real models, DO NOT run systemctl --user enable/ daemon-reload. Artifacts + tests only.
