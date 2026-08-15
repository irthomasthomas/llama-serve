#!/usr/bin/env bash
# ============================================================
# llama-serve-v2.sh — Modernized multi-model llama-server launcher
# env-wrapper (gpu-guard shim fix), speculative decoding via registry draft= field.
# One server per model family, best quant, per-model tuning.
# ============================================================
set -euo pipefail

LLAMA_BIN="${LLAMA_BIN:-/home/thomas/llama.cpp/build_optimized/bin/llama-server}"
BEE_BIN="${BEE_BIN:-/home/thomas/beellama.cpp/build/bin/llama-server}"
BEE_LIB_DIR="/home/thomas/beellama.cpp/build/bin"
MODELS_DIR="/home/thomas/models"
LOG_DIR="${LLAMA_LOG_DIR:-/tmp/llama_logs}"
PID_DIR="${LLAMA_PID_DIR:-/tmp/llama_pids}"
mkdir -p "$LOG_DIR" "$PID_DIR"

# ---- Model registry ----------------------------------------
# Each entry: "file|port|ctx|ngl|extra_args|draft=<path-or-empty>"
# Field 6 (draft=...) selects a draft model for speculative decoding.
# Categories inferred from flags: --embedding => embedding, etc.
declare -A REGISTRY=(
  # === Main chat models ===
  [lfm-8b-base]="$MODELS_DIR/8B-A1B-Base-GGUF/LFM2.5-8B-A1B-Base-Q6_K.gguf|8090|128000|99|--alias lfm2.5-8b-a1b-base --temp 0.2 --top-k 80 --repeat-penalty 1.05|draft="
  [lfm-8b]="$MODELS_DIR/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q6_K.gguf|8080|128000|99|--alias lfm2.5-8b-a1b --temp 0.2 --top-k 80 --repeat-penalty 1.05 --chat-template-kwargs {\"keep_past_thinking\":true} --spec-type ngram-mod|draft=" # draft= unused: no same-vocab LFM2.5 draft exists; ngram-simple self-speculation instead (b9139, ~45%% accept on repetitive text, +8 tok/s, 0 extra VRAM)
  # [lfm-8b-maxctx]="$MODELS_DIR/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf|8080|128000|99|--alias lfm2.5-8b-a1b-128k --chat-template-kwargs {\"keep_past_thinking\":true} --top-k 80 --repeat-penalty 1.05"
  [lfm-1.2b]="$MODELS_DIR/LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q4_K_M.gguf|8081|128000|99|--alias lfm2.5-1.2b-instruct|draft="
  # LFM2.5-2.6B: uses BeeLlama fork (5% faster gen than official at Q8_0)
  # Benchmarks (RTX 3060): official 98.5 tok/s, BeeLlama 103.8 tok/s, BeeLlama IQ4_XS 144.8 tok/s
  [lfm-2.6b]="$MODELS_DIR/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q8_0.gguf|8087|65536|99|--alias lfm2.5-2.6b --temp 0.1 --top-k 50 --repeat-penalty 1.1|draft=|bin=$BEE_BIN"
  # Low-VRAM fallback (2.6 GB VRAM, 3.4 GB total) — BeeLlama only, edge of quality cliff
  [lfm-2.6b-lowmem]="$MODELS_DIR/LFM2.5-2.6B-IQ4_XS/LiquidAI_LFM2.5-2.6B-IQ4_XS.gguf|8087|65536|99|--alias lfm2.5-2.6b-lowmem --temp 0.2 --top-k 80 --repeat-penalty 1.05 --cache-type-k q3_0 --cache-type-v q3_0 --kv-tail-tokens 512|draft=|bin=$BEE_BIN"
  # 128K context — LFM Mamba hybrid makes this cheap: Q8_0=4.6 GB, IQ4_XS=2.6 GB VRAM
  [lfm-2.6b-128k]="$MODELS_DIR/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q8_0.gguf|8087|131072|99|--alias lfm2.5-2.6b-128k --temp 0.2 --top-k 80 --repeat-penalty 1.05|draft=|bin=$BEE_BIN"
  # 128K low-VRAM: 2.6 GB total — fits alongside lfm-8b in 12 GB cards
  [lfm-2.6b-128k-lowmem]="$MODELS_DIR/LFM2.5-2.6B-IQ4_XS/LiquidAI_LFM2.5-2.6B-IQ4_XS.gguf|8087|131072|99|--alias lfm2.5-2.6b-128k-lowmem --temp 0.2 --top-k 80 --repeat-penalty 1.05 --cache-type-k q3_0 --cache-type-v q3_0 --kv-tail-tokens 512|draft=|bin=$BEE_BIN"
  [vibethinker-3b]="$MODELS_DIR/VibeThinker-3B-GGUF/VibeThinker-3B.Q5_K_M.gguf|8082|131072|99|--alias vibethinker-3b --reasoning auto|draft="
  [fastcontext-rl]="$MODELS_DIR/FastContext-1.0-4B-RL-GGUF/FastContext-1.0-4B-RL.Q6_K.gguf|8083|65536|99|--alias fastcontext-4b-rl|draft="
  [fastcontext-sft]="$MODELS_DIR/FastContext-1.0-4B-SFT-Q5_K_M/fastcontext-1.0-4b-sft-q5_k_m.gguf|8084|65536|99|--alias fastcontext-4b-sft|draft="

  # === Embedding models ===
  [lfm-embedding]="$MODELS_DIR/Embedding-350M-GGUF/LFM2.5-Embedding-350M-F16.gguf|8085|8192|99|--alias lfm2.5-embedding-350m --embedding --pooling mean|draft="
  [lfm-colbert]="$MODELS_DIR/ColBERT-350M-GGUF/LFM2.5-ColBERT-350M-F16.gguf|8086|8192|99|--alias lfm2-colbert-350m --rerank --pooling rank|draft="

  # === Vision-language models ===
  # LFM2.5-VL-3B benchmarks (RTX 3060, 280-tok gen): Q8_0 94.4 tok/s / ~3.3 GB VRAM, Q6_K 110.1 tok/s / ~2.7 GB VRAM
  [lfm2-vl-450m]="$MODELS_DIR/LFM2.5-VL-450M-GGUF/LFM2.5-VL-450M-Q8_0.gguf|8088|8192|99|--alias lfm2.5-vl-450m --mmproj $MODELS_DIR/LFM2.5-VL-450M-GGUF/mmproj-LFM2.5-VL-450m-Q8_0.gguf|draft="
  [lfm2-vl-1.6b]="/home/thomas/.cache/huggingface/hub/models--LiquidAI--LFM2.5-VL-1.6B-GGUF/snapshots/0df8719db7180cedababc2bc589abfe5e8ebcd1f/LFM2.5-VL-1.6B-Q8_0.gguf|8089|32768|99|--alias lfm2.5-vl-1.6b --mmproj /home/thomas/.cache/huggingface/hub/models--LiquidAI--LFM2.5-VL-1.6B-GGUF/snapshots/0df8719db7180cedababc2bc589abfe5e8ebcd1f/mmproj-LFM2.5-VL-1.6b-Q8_0.gguf|draft="
  [lfm2-vl-3b-8bit]="$MODELS_DIR/LFM2.5-VL-3B-GGUF/LFM2.5-VL-3B-Q8_0.gguf|8093|65536|99|--alias lfm2.5-vl-3b-8bit --mmproj $MODELS_DIR/LFM2.5-VL-3B-GGUF/mmproj-LFM2.5-VL-3B-Q8_0.gguf|draft="
  [lfm2-vl-3b-6bit]="$MODELS_DIR/LFM2.5-VL-3B-GGUF/LFM2.5-VL-3B-Q6_K.gguf|8094|65536|99|--alias lfm2.5-vl-3b-6bit --mmproj $MODELS_DIR/LFM2.5-VL-3B-GGUF/mmproj-LFM2.5-VL-3B-Q8_0.gguf|draft="
  # === vLLM-served models (non-GGUF; extra="vllm:<launcher>" delegates to external script) ===
  [ling-3.0-tiny]="/home/thomas/models/Ling-3.0-tiny-int4|8300|8192|0|vllm:/home/thomas/ling3-start.sh|draft=|bin="
)

# ---- Helpers -----------------------------------------------
_get_field() { local entry="$1" idx="$2"; echo "$entry" | cut -d'|' -f"$idx"; }
_pid_file()  { echo "$PID_DIR/$1.pid"; }
_log_file()  { echo "$LOG_DIR/$1.log"; }

_is_running() {
  local pf="$1"
  [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null
}

# ---- VRAM measurement (multi-tier fallback) ----------------
# NVML breaks whenever the kernel module / userspace lib versions drift
# ("Failed to initialize NVML: Driver/library version mismatch"), so try:
#   1. nvidia-smi (NVML)
#   2. sysfs  mem_info_vram_{total,used}  (amdgpu / nouveau; NVIDIA lacks it)
#   3. total from /proc/driver/nvidia/gpus/*/information + model lookup,
#      used summed from llama-server logs (exact self-reported CUDA bufs)
#
# GPU total MiB lookup by marketing name (extend as needed).
_gpu_total_lookup() {
  local model="$1"
  case "$model" in
    *"RTX 3060 Ti"*)           echo 8192  ;;
    *"RTX 3060"*)              echo 12288 ;;
    *"RTX 3070 Ti"*)           echo 8192  ;;
    *"RTX 3070"*)              echo 8192  ;;
    *"RTX 3080 Ti"*)           echo 12288 ;;
    *"RTX 3080 12GB"*)         echo 12288 ;;
    *"RTX 3080"*)              echo 10240 ;;
    *"RTX 3090 Ti"*)           echo 24576 ;;
    *"RTX 3090"*)              echo 24576 ;;
    *"RTX 4060 Ti 16GB"*)      echo 16384 ;;
    *"RTX 4060 Ti"*)           echo 8192  ;;
    *"RTX 4060"*)              echo 8192  ;;
    *"RTX 4070 Ti SUPER"*)     echo 16384 ;;
    *"RTX 4070 Ti"*)           echo 12288 ;;
    *"RTX 4070 SUPER"*)        echo 12288 ;;
    *"RTX 4070"*)              echo 12288 ;;
    *"RTX 4080 SUPER"*)        echo 16384 ;;
    *"RTX 4080"*)              echo 16384 ;;
    *"RTX 4090"*)              echo 24576 ;;
    *"RTX 5070 Ti"*)           echo 16384 ;;
    *"RTX 5070"*)              echo 12288 ;;
    *"RTX 5080"*)              echo 16384 ;;
    *"RTX 5090"*)              echo 32768 ;;
    *"A100 80GB"*)             echo 81920 ;;
    *"A100"*)                  echo 40960 ;;
    *"H100"*)                  echo 81920 ;;
    *"L40"*)                   echo 49152 ;;
    *"A10"*)                   echo 24576 ;;
    *"T4"*)                    echo 16384 ;;
    *"V100"*)                  echo 16384 ;;
    *)                         echo 0     ;;
  esac
}

# Total VRAM MiB from /proc (works without NVML)
_gpu_total_proc() {
  local f total=0 m
  for f in /proc/driver/nvidia/gpus/*/information; do
    [[ -r "$f" ]] || continue
    m=$(grep -m1 '^Model:' "$f" 2>/dev/null | cut -d: -f2-)
    [[ -n "$m" ]] && total=$(( total + $(_gpu_total_lookup "$m") ))
  done
  echo "$total"
}

# Sum of CUDA buffers a llama-server reported in its own log (exact MiB).
# Latest occurrence per buffer class wins (restarts rewrite the log).
_llama_log_vram_mib() {
  local lf="$1"
  [[ -r "$lf" ]] || { echo 0; return; }
  tac "$lf" 2>/dev/null | awk '
    /CUDA[0-9]+ model buffer size/ && !m { m=$(NF-1) }
    /KV buffer size/               && !k { k=$(NF-1) }
    /output buffer size/           && !o { o=$(NF-1) }
    /compute buffer size/          && !c { c=$(NF-1) }
    m && k { printf "%d", int(m+k+o+c+0.5); exit }
  ' 2>/dev/null || echo 0
}

# Used MiB estimate when NVML is dead: tracked running models' self-reported
# CUDA buffers + a desktop fudge factor.
_vram_used_estimate() {
  local used=0 name pf
  for name in "${!REGISTRY[@]}"; do
    pf="$(_pid_file "$name")"
    _is_running "$pf" || continue
    used=$(( used + $(_llama_log_vram_mib "$(_log_file "$name")") ))
  done
  if (( used > 0 )); then echo $(( used + 500 )); else echo 0; fi
}

# _vram_totals:  echoes "<used_mib> <total_mib>" or "" if unknown.
_vram_totals() {
  # --- tier 1: NVML ---
  local out
  if out=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null); then
    local used total
    used=$(awk -F', ' '{s+=$1} END{print s+0}' <<<"$out")
    total=$(awk -F', ' '{s+=$2} END{print s+0}' <<<"$out")
    if [[ -n "$used" && -n "$total" && "$total" -gt 0 ]]; then
      echo "$used $total"; return 0
    fi
  fi

  # --- tier 2: sysfs (bytes -> MiB) ---
  local t=0 u=0 f
  for f in /sys/bus/pci/devices/*/mem_info_vram_total \
           /sys/class/drm/card*/device/mem_info_vram_total; do
    [[ -r "$f" ]] || continue
    t=$(( t + $(cat "$f") ))
    local uf="${f%_total}_used"
    [[ -r "$uf" ]] && u=$(( u + $(cat "$uf") ))
  done
  if (( t > 0 )); then
    echo "$(( u / 1048576 )) $(( t / 1048576 ))"; return 0
  fi

  # --- tier 3: proc total + log-based used estimate ---
  local pt
  pt=$(_gpu_total_proc)
  if (( pt > 0 )); then
    echo "$(_vram_used_estimate) $pt"; return 0
  fi

  return 1
}

# Free MiB (unknown => -1)
_vram_free_mib() {
  local totals
  if totals=$(_vram_totals); then
    local used total
    read -r used total <<<"$totals"
    if (( total > 0 )); then echo $(( total - used )); else echo "-1"; fi
  else
    echo "-1"
  fi
}

# Human-readable one-line summary
_vram_nvml_ok() {
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits >/dev/null 2>&1
}

_vram_summary() {
  local totals
  if totals=$(_vram_totals); then
    local used total
    read -r used total <<<"$totals"
    if (( total > 0 )); then
      if _vram_nvml_ok; then
        echo "${used} MiB used / ${total} MiB total ($(( total - used )) MiB free)"
      else
        echo "~${used} MiB used / ${total} MiB total (~$(( total - used )) MiB free) [estimated: NVML broken]"
      fi
    else
      echo "~${used} MiB used (total unknown — NVML unavailable, sysfs total missing)"
    fi
  else
    echo "unknown (NVML unavailable and no VRAM counters found)"
  fi
}

# PIDs of every process holding an NVIDIA device fd
_gpu_holders() {
  local p fd
  for p in /proc/[0-9]*; do
    for fd in "$p"/fd/*; do
      [[ -e "$fd" ]] || continue
      if [[ "$(readlink "$fd" 2>/dev/null)" =~ ^/dev/nvidia[0-9] ]]; then
        basename "$p"
        break
      fi
    done
  done | sort -un
}

# Per-process VRAM estimate via nvidia-caps:  "pid mib" per line
_gpu_holders_vram() {
  local pid
  for pid in $(_gpu_holders); do
    local mib=0 cap_name mf
    for cap_name in /proc/driver/nvidia/capabilities/*/*; do
      mf="$cap_name/MiB"
      [[ -r "$mf" ]] || continue
      # only count caps this pid actually holds
      local fd held=0
      for fd in /proc/$pid/fd/*; do
        [[ -e "$fd" ]] || continue
        [[ "$(readlink "$fd" 2>/dev/null)" == "$cap_name" ]] && { held=1; break; }
      done
      (( held )) && mib=$(( mib + $(awk '{print $1}' "$mf" 2>/dev/null || echo 0) ))
    done
    echo "$pid $mib"
  done
}

_est_weight_gb() {
  # Rough estimate from file size
  local f="$1"
  local sz
  sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
  echo "scale=2; $sz / 1073741824" | bc 2>/dev/null || echo "?"
}

# ---- Commands ----------------------------------------------
start_model() {
  local name="$1"
  local entry="${REGISTRY[$name]:-}"
  [[ -z "$entry" ]] && { echo "[ERR] Unknown model: $name"; echo "Available: ${!REGISTRY[*]}"; return 1; }

  local file port ctx ngl extra draft
  file="$(_get_field "$entry" 1)"
  port="$(_get_field "$entry" 2)"
  ctx="$(_get_field "$entry" 3)"
  ngl="$(_get_field "$entry" 4)"
  extra="$(_get_field "$entry" 5)"
  draft="$(_get_field "$entry" 6)"
  draft="${draft#draft=}"   # strip leading "draft="

  local model_bin
  model_bin="$(_get_field "$entry" 7)"
  model_bin="${model_bin#bin=}"
  [[ -z "$model_bin" ]] && model_bin="$LLAMA_BIN"

  # Cache-type override: skip hardcoded defaults if extra args specify them
  local ctk_args="-ctk q8_0 -ctv q8_0"
  if echo "$extra" | grep -q -- '--cache-type-k\|--ctk'; then
    ctk_args=""
  fi

  # LD_LIBRARY_PATH for BeeLlama shared libs
  local model_ld_path=""
  if [[ "$model_bin" == "$BEE_BIN" ]]; then
    model_ld_path="$BEE_LIB_DIR"
  fi

  # vLLM-served model: delegate to external launcher script (port readiness poll)
  if [[ "$extra" == vllm:* ]]; then
    local launcher="${extra#vllm:}"
    launcher="${launcher/#\~/$HOME}"
    pf="$(_pid_file "$name")"
    if _is_running "$pf"; then
      echo "[OK] $name already running (PID $(cat "$pf")) on port $port"
      return 0
    fi
    [[ ! -x "$launcher" ]] && { echo "[ERR] Launcher not executable: $launcher"; return 1; }
    local lf="$(_log_file "$name")"
    echo "[START] $name via vLLM launcher: $launcher (log: $lf)"
    nohup "$launcher" > "$lf" 2>&1 &
    local vpid=$!
    echo "$vpid" > "$pf"
    disown "$vpid" 2>/dev/null || true
    local i
    for i in $(seq 1 120); do
      curl -sf -m 2 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && break
      _is_running "$pf" || { echo "[ERR] $name launcher died — see $lf"; tail -5 "$lf" 2>/dev/null; rm -f "$pf"; return 1; }
      sleep 5
    done
    curl -sf -m 2 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1       && echo "[OK] $name serving on port $port"       || echo "[WARN] $name not answering on :$port after 10min; still loading? log: $lf"
    return 0
  fi

  [[ ! -f "$file" ]] && { echo "[ERR] File not found: $file"; return 1; }

  local pf

  pf="$(_pid_file "$name")"
  if _is_running "$pf"; then
    echo "[OK] $name already running (PID $(cat "$pf")) on port $port"
    return 0
  fi

  # VRAM heads-up for GPU models
  if [[ "$ngl" -gt 0 ]]; then
    local free_mib wgb
    free_mib=$(_vram_free_mib)
    wgb=$(_est_weight_gb "$file")
    local free_str
    if [[ "$free_mib" == "-1" ]]; then
      free_str="unknown ($(_vram_summary))"
    else
      free_str="${free_mib}MiB"
    fi
    echo "[INFO] $name: weights ~${wgb}GB, VRAM free ${free_str}, ctx=$ctx ngl=$ngl bin=$(basename "$model_bin")"
  fi

  # Kill any stale process on this port
  if ss -tlnp 2>/dev/null | grep -q ":${port}"; then
    echo "[WARN] Port $port busy, attempting to free..."
    local oldpid
    oldpid=$(ss -tlnp 2>/dev/null | grep ":${port}" | grep -oP 'pid=\K[0-9]+' | head -1)
    [[ -n "$oldpid" ]] && kill "$oldpid" 2>/dev/null || true
    sleep 1
  fi

  # Speculative-decoding flags (only when a draft model is configured)
  local spec_flags=""
  if [[ -n "$draft" ]]; then
    spec_flags="--model-draft $draft --spec-draft-ngl 99 --spec-draft-n-max 8 --spec-draft-n-min 2"
    echo "[INFO] $name: speculative decoding enabled (draft: $(basename "$draft"))"
  fi

  local lf

  lf="$(_log_file "$name")"
  echo "[START] $name -> port $port (log: $lf)"

  # env wrapper strips gpu-guard shim (LD_PRELOAD) + hidden-GPU (CUDA_VISIBLE_DEVICES="")
  # that agent shells export; without it llama-server silently runs CPU-only. FACTS.md.
  # shellcheck disable=SC2086
  nohup env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0 ${model_ld_path:+LD_LIBRARY_PATH="$model_ld_path"} "$model_bin" \
    -m "$file" \
    -ngl "$ngl" \
    -c "$ctx" \
    --host 0.0.0.0 \
    --port "$port" \
    -t 8 \
    -tb 16 \
    -cb \
    -np 1 \
    -fa on \
    $ctk_args \
    --jinja \
    --cache-reuse 256 \
    $spec_flags \
    $extra \
    > "$lf" 2>&1 &
  local pid=$!
  echo "$pid" > "$pf"
  disown "$pid" 2>/dev/null || true

  # Wait briefly and check it didn't immediately die
  sleep 2
  if _is_running "$pf"; then
    echo "[OK] $name started (PID $pid) on port $port"
  else
    echo "[ERR] $name died immediately — check $lf"
    tail -5 "$lf" 2>/dev/null
    rm -f "$pf"
    return 1
  fi
}

stop_model() {
  local name="$1"
  local pf
  pf="$(_pid_file "$name")"
  if _is_running "$pf"; then
    local pid
    pid=$(cat "$pf")
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
    echo "[STOP] $name (PID $pid) stopped"
  else
    echo "[INFO] $name not running"
  fi
  rm -f "$pf"
}

status_all() {
  printf "%-22s %-8s %-15s %-8s %-8s %-10s %s\n" "MODEL" "PORT" "STATUS" "CTX" "NGL" "VRAM(MiB)" "FILE"
  printf "%-22s %-8s %-15s %-8s %-8s %-10s %s\n" "------" "----" "------" "----" "----" "----------" "----"
  local name
  for name in $(echo "${!REGISTRY[@]}" | tr ' ' '\n' | sort); do
    local entry="${REGISTRY[$name]}"
    local port ctx ngl file
    file="$(_get_field "$entry" 1)"
    port="$(_get_field "$entry" 2)"
    ctx="$(_get_field "$entry" 3)"
    ngl="$(_get_field "$entry" 4)"
    local pf
    pf="$(_pid_file "$name")"
    local st="stopped" vm="-"
    if _is_running "$pf"; then
      st="running:$(cat "$pf")"
      vm=$(_llama_log_vram_mib "$(_log_file "$name")")
    fi
    printf "%-22s %-8s %-15s %-8s %-8s %-10s %s\n" "$name" "$port" "$st" "$ctx" "$ngl" "$vm" "$(basename "$file")"
  done
  echo
  echo "VRAM: $(_vram_summary)"
}

tail_log() {
  local name="${1:-}"
  if [[ -z "$name" ]]; then
    echo "Usage: $0 log <model-name>"
    echo "Available: ${!REGISTRY[*]}"
    return 1
  fi
  tail -f "$(_log_file "$name")"
}

list_models() {
  echo "Available models (one per family, best quant):"
  echo
  local name
  for name in $(echo "${!REGISTRY[@]}" | tr ' ' '
' | sort); do
    local entry="${REGISTRY[$name]}"
    local file port ctx
    file="$(_get_field "$entry" 1)"
    port="$(_get_field "$entry" 2)"
    ctx="$(_get_field "$entry" 3)"
    local wgb
    wgb=$(_est_weight_gb "$file")
    printf "  %-22s port=%-5s ctx=%-7s ~%sGB  %s
" "$name" "$port" "$ctx" "$wgb" "$(basename "$file")"
  done
}

stop_all() {
  local name
  for name in "${!REGISTRY[@]}"; do stop_model "$name"; done
}

# ---- VRAM management ----------------------------------------
vram_report() {
  echo "== VRAM =="
  echo "  $(_vram_summary)"
  if ! _vram_nvml_ok; then
    echo "  [WARN] NVML unavailable (driver/library mismatch?) — used figure is log-based estimate"
    echo "         fix: match kernel module & userspace versions, or reboot after driver update"
  fi
  echo
  echo "== Tracked models (self-reported CUDA buffers) =="
  local name pf vm any=0
  for name in $(echo "${!REGISTRY[@]}" | tr ' ' '\n' | sort); do
    pf="$(_pid_file "$name")"
    _is_running "$pf" || continue
    any=1
    vm=$(_llama_log_vram_mib "$(_log_file "$name")")
    printf "  %-22s PID %-9s ~%s MiB\n" "$name" "$(cat "$pf")" "$vm"
  done
  (( any )) || echo "  (none running)"
  echo
  echo "== Processes holding the GPU =="
  local pid mib
  local found=0
  while read -r pid mib; do
    [[ -n "$pid" ]] || continue
    found=1
    local cmd
    cmd=$(ps -o comm= -p "$pid" 2>/dev/null || echo "?")
    if (( mib > 0 )); then
      printf "  %-8s %-10s %s MiB\n" "$pid" "$cmd" "$mib"
    else
      printf "  %-8s %-10s (vram n/a)\n" "$pid" "$cmd"
    fi
  done < <(_gpu_holders_vram)
  (( found )) || echo "  (none)"
}

clear_vram() {
  # Usage: clear [--llama-only] [--yes]
  # Kills every process holding /dev/nvidiaN (default) or just llama-servers
  # (--llama-only). Stops tracked models first via pid files.
  local llama_only=0 assume_yes=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --llama-only|-l) llama_only=1 ;;
      --yes|-y)        assume_yes=1 ;;
      *) echo "[ERR] clear: unknown flag $1"; return 1 ;;
    esac
    shift
  done

  # 1. Stop all tracked models cleanly (removes stale pid files)
  stop_all

  # 2. Find remaining GPU holders
  local pids=()
  local pid
  for pid in $(_gpu_holders); do
    if (( llama_only )); then
      local cmd
      cmd=$(ps -o comm= -p "$pid" 2>/dev/null || true)
      [[ "$cmd" == *llama* ]] || continue
    fi
    # never kill ourselves or our ancestors
    if [[ "$pid" == "$$" ]] || grep -qw "$pid" <<<"$(pstree -ps $$ 2>/dev/null | grep -oP '\(\K[0-9]+(?=\))' | tr '
' ' ')"; then
      continue
    fi
    pids+=("$pid")
  done

  if (( ${#pids[@]} == 0 )); then
    echo "[OK] No GPU processes to kill. VRAM: $(_vram_summary)"
    return 0
  fi

  echo "[INFO] Processes holding the GPU:"
  for pid in "${pids[@]}"; do
    printf "  %-8s %s
" "$pid" "$(ps -o args= -p "$pid" 2>/dev/null | cut -c1-100 || echo '?')"
  done

  if (( ! assume_yes )); then
    local ans
    read -r -p "Kill these ${#pids[@]} process(es)? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || { echo "[ABORT]"; return 1; }
  fi

  for pid in "${pids[@]}"; do
    echo "[KILL] $pid ($(ps -o comm= -p "$pid" 2>/dev/null || echo '?'))"
    kill "$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in "${pids[@]}"; do
    kill -9 "$pid" 2>/dev/null || true
  done
  sleep 1
  echo "[OK] VRAM cleared. Now: $(_vram_summary)"
}

evict_models() {
  # Usage: evict [name ...]        stop specific models (same as stop)
  #        evict --all             stop every tracked model
  #        evict --interactive|-i  pick running models to stop
  #        evict --free <MiB>      stop models (largest ctx first) until
  #                                at least <MiB> free, or all stopped
  local interactive=0 free_target="" names=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --all|-a)        names=("${!REGISTRY[@]}") ;;
      --interactive|-i) interactive=1 ;;
      --free)          free_target="${2:?--free needs MiB}"; shift ;;
      *)               names+=("$1") ;;
    esac
    shift
  done

  if (( interactive )); then
    local running=() name pf
    for name in $(echo "${!REGISTRY[@]}" | tr ' ' '
' | sort); do
      pf="$(_pid_file "$name")"
      _is_running "$pf" && running+=("$name")
    done
    if (( ${#running[@]} == 0 )); then
      echo "[INFO] No tracked models running."
      return 0
    fi
    echo "Running models:"
    local i
    for i in "${!running[@]}"; do
      echo "  $((i+1))) ${running[$i]} (PID $(cat "$(_pid_file "${running[$i]}")"))"
    done
    local sel
    read -r -p "Numbers to evict (space-separated, 'a' = all): " sel
    [[ "$sel" == "a" ]] && names=("${running[@]}") || {
      for i in $sel; do
        [[ "$i" =~ ^[0-9]+$ ]] && (( i >= 1 && i <= ${#running[@]} )) \
          && names+=("${running[$((i-1))]}")
      done
    }
  fi

  if [[ -n "$free_target" ]]; then
    # Evict running models (largest ctx first) until target reached
    local free_now
    free_now=$(_vram_free_mib)
    if [[ "$free_now" != "-1" && "$free_now" -ge "$free_target" ]]; then
      echo "[OK] Already ${free_now} MiB free (target $free_target)."
      return 0
    fi
    # sort running models by ctx descending
    local sorted
    sorted=$(
      for name in "${!REGISTRY[@]}"; do
        local pf
        pf="$(_pid_file "$name")"
        _is_running "$pf" || continue
        echo "$(_get_field "${REGISTRY[$name]}" 3) $name"
      done | sort -rn
    )
    if [[ -z "$sorted" ]]; then
      echo "[INFO] No tracked models running; cannot free more by eviction."
      echo "       Try: $0 clear   (kills untracked GPU processes too)"
      return 1
    fi
    while read -r _ name; do
      free_now=$(_vram_free_mib)
      if [[ "$free_now" != "-1" && "$free_now" -ge "$free_target" ]]; then
        echo "[OK] ${free_now} MiB free (target $free_target)."
        return 0
      fi
      stop_model "$name"
      sleep 1
    done <<<"$sorted"
    free_now=$(_vram_free_mib)
    echo "[DONE] Evicted all running models. Free now: ${free_now} MiB"
    return 0
  fi

  if (( ${#names[@]} == 0 )); then
    echo "Usage: $0 evict [--all|--interactive|--free <MiB>|name ...]"
    return 1
  fi
  local name
  for name in "${names[@]}"; do stop_model "$name"; done
}

# ---- Main dispatch ------------------------------------------
case "${1:-}" in
  start)
    shift
    [[ -z "${1:-}" ]] && { echo "Usage: $0 start <model-name> [model-name...]"; exit 1; }
    for m in "$@"; do start_model "$m" || true; done
    ;;
  stop)
    shift
    if [[ -z "${1:-}" ]]; then stop_all; else
      for m in "$@"; do stop_model "$m"; done
    fi
    ;;
  status|st) status_all ;;
  list|ls)   list_models ;;
  log|logs)  shift; tail_log "$@" ;;
  vram)      vram_report ;;
  clear)     shift; clear_vram "$@" ;;
  evict)     shift; evict_models "$@" ;;
  restart)
    shift; for m in "$@"; do stop_model "$m"; start_model "$m"; done
    ;;
  ""|help|-h|--help)
    cat <<EOF
Usage: $0 <command> [args]

Commands:
  list                     Show all registered models
  status                   Show running state of all models (+ VRAM)
  start <name> [name...]   Start one or more models
  stop  <name> [name...]   Stop specific models (or all if none given)
  restart <name> [name...] Restart models
  log <name>               Tail a model's log

VRAM management:
  vram                     VRAM usage + per-process GPU holders
  evict --interactive|-i   Interactively pick running models to stop
  evict --free <MiB>       Stop models (largest ctx first) until MiB free
  evict --all|<name...>    Stop all/specific models
  clear [--llama-only] [-y]  Kill ALL processes holding the GPU
                           (stops tracked models first; -y skips confirm)

VRAM measurement works even when NVML is broken (driver/library mismatch):
falls back to sysfs /proc accounting automatically.

Models are addressed by short name (see: $0 list).
Each runs on its own port with --alias for OpenAI-compatible naming.
EOF
    ;;
  *) echo "Unknown command: $1"; exit 1 ;;
esac
