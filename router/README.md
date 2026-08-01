# llama-server router mode

A **single OpenAI-compatible endpoint** on `:8080` serves every chat and
vision-language model. The router (llama.cpp b9139, `--models-preset`) loads
model instances on demand and forwards each request to the right one based on
the `"model"` field in the request body (POST) or the `?model=` query param
(GET). Models are defined in [`presets.ini`](./presets.ini); operational knobs
live in [`serve.conf`](./serve.conf).

## How to run

```sh
LLAMA_BIN=/home/thomas/llama.cpp/build_optimized/bin/llama-server
# Strip the GPU-guard env that silently forces CPU (see FACTS.md).
env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0 "$LLAMA_BIN" \
    --host 0.0.0.0 --port 8080 \
    --models-preset /home/thomas/work/llama-serve-modern/router/presets.ini \
    --models-max 2 \
    --sleep-idle-seconds 300
```

Chat models are **not** all loaded at once. `--models-max 2` keeps at most two
chat/VL instances resident; the rest are loaded on first request and put to
sleep after `--sleep-idle-seconds 300` of idleness. Tiny always-on services
(embedding, reranker) are kept **outside** the router so they are never
evicted.

## Always-on standalone services

These ports stay resident regardless of router activity (tiny models, zero
cold-start):

| Port  | Service              | Model (router `model` field)  |
|-------|----------------------|-------------------------------|
| 8085  | Embeddings           | `lfm2.5-embedding-350m`       |
| 8086  | Reranker (ColBERT)   | `lfm2-colbert-350m`            |

The router on `:8080` can also serve these (they are defined in `presets.ini`),
but in production they run as standalone `llama-server` processes so the
router's `MODELS_MAX` budget never evicts them.

## Single chat/VL endpoint

Send chat completions to `http://localhost:8080/v1/chat/completions` and pick
the model with the `model` field:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"lfm2.5-8b-a1b","messages":[{"role":"user","content":"hi"}]}'
```

Vision works the same way with `lfm2.5-vl-450m` / `lfm2.5-vl-1.6b` and an
`image_url` content part.

## Routing keys

Each `[section]` in `presets.ini` is a routing key, and each section also sets
`alias = ...` so the OpenAI-style name resolves too. Either form works:

| `model` value (any of these)        | Section         | File                                 |
|-------------------------------------|-----------------|--------------------------------------|
| `lfm-8b` / `lfm2.5-8b-a1b`          | `[lfm-8b]`      | LFM2.5-8B-A1B-Q6_K.gguf (+1.2B draft)|
| `lfm-8b-base` / `lfm2.5-8b-a1b-base`| `[lfm-8b-base]` | LFM2.5-8B-A1B-Base-Q6_K.gguf         |
| `lfm-1.2b` / `lfm2.5-1.2b-instruct` | `[lfm-1.2b]`    | LFM2.5-1.2B-Instruct-Q4_K_M.gguf     |
| `vibethinker-3b`                    | `[vibethinker-3b]` | VibeThinker-3B.Q5_K_M.gguf        |
| `fastcontext-rl` / `fastcontext-4b-rl`  | `[fastcontext-rl]`  | FastContext-1.0-4B-RL.Q6_K.gguf  |
| `fastcontext-sft` / `fastcontext-4b-sft`| `[fastcontext-sft]` | fastcontext-1.0-4b-sft-q5_k_m.gguf |
| `lfm2-vl-450m` / `lfm2.5-vl-450m`   | `[lfm2-vl-450m]`| LFM2.5-VL-450M-Q8_0.gguf            |
| `lfm2-vl-1.6b` / `lfm2.5-vl-1.6b`   | `[lfm2-vl-1.6b]`| LFM2.5-VL-1.6B-Q8_0.gguf            |

## Legacy port → router migration

Before this change every model had its own port. Clients should move to the
single router endpoint and select via `model`. The old ports can be kept as
thin redirects/standby, but the router is now the source of truth for chat/VL.

| Legacy port | Legacy model    | New access                                            |
|-------------|-----------------|-------------------------------------------------------|
| 8080        | lfm-8b          | **router :8080**, `model=lfm2.5-8b-a1b` (unchanged)   |
| 8081        | lfm-1.2b        | router :8080, `model=lfm2.5-1.2b-instruct`            |
| 8082        | vibethinker-3b  | router :8080, `model=vibethinker-3b`                  |
| 8083        | fastcontext-rl  | router :8080, `model=fastcontext-4b-rl` (FastContext service may also stay on 8083) |
| 8084        | fastcontext-sft | router :8080, `model=fastcontext-4b-sft`              |
| 8085        | lfm-embedding   | **standalone :8085** (unchanged)                      |
| 8086        | lfm-colbert     | **standalone :8086** (unchanged)                      |
| 8088        | lfm2-vl-450m    | router :8080, `model=lfm2.5-vl-450m`                  |
| 8089        | lfm2-vl-1.6b    | router :8080, `model=lfm2.5-vl-1.6b`                  |
| 8090        | lfm-8b-base     | router :8080, `model=lfm2.5-8b-a1b-base`              |

## Telegram bot contract

The Telegram bot must point at the **router** and select the chat model with the
`model` field:

- **Endpoint:** `http://localhost:8080/v1/chat/completions`
- **Model field:** `lfm2.5-8b-a1b`

```json
{"model": "lfm2.5-8b-a1b", "messages": [...]}
```

`lfm2.5-8b-a1b` is the `alias` of the `[lfm-8b]` preset (the 8B-A1B Q6_K model
with 1.2B speculative decoding), so it resolves on the router. Do not address
the bot at `:8081` or any per-model port — those are legacy.
