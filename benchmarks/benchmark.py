#!/usr/bin/env python3
"""Systematic inference speed benchmarks for local GGUF models."""
import json, os, subprocess, sys, time, urllib.request, sqlite3, argparse
from pathlib import Path

DB_PATH = Path(__file__).parent / "perf_benchmarks.db"
OFFICIAL_BIN = "/home/thomas/llama.cpp/build_optimized/bin/llama-server"
BEE_BIN = "/home/thomas/bin/llama-server-beellama"
OFFICIAL_LIB = "/home/thomas/llama.cpp/build_optimized/bin"
PORT = 18099

MODELS = {
    "lfm-2.6b": {"file": "/home/thomas/models/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q8_0.gguf",
        "arch": "lfm2-hybrid", "params": "2.6B", "quant": "Q8_0", "binary": "bee",
        "kv": ["q8_0","q8_0"], "extra": ""},
    "lfm-2.6b-iq4": {"file": "/home/thomas/models/LFM2.5-2.6B-IQ4_XS/LiquidAI_LFM2.5-2.6B-IQ4_XS.gguf",
        "arch": "lfm2-hybrid", "params": "2.6B", "quant": "IQ4_XS", "binary": "bee",
        "kv": ["q3_0","q3_0"], "extra": "--kv-tail-tokens 512"},
    "lfm-1.2b": {"file": "/home/thomas/models/LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
        "arch": "lfm2-hybrid", "params": "1.2B", "quant": "Q4_K_M", "binary": "official",
        "kv": ["q8_0","q8_0"], "extra": ""},
    "lfm-8b": {"file": "/home/thomas/models/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q6_K.gguf",
        "arch": "lfm2-hybrid", "params": "8B-A1B", "quant": "Q6_K", "binary": "official",
        "kv": ["q8_0","q8_0"], "extra": "--spec-type ngram-mod"},
    "vibethinker-3b": {"file": "/home/thomas/models/VibeThinker-3B-GGUF/VibeThinker-3B.Q5_K_M.gguf",
        "arch": "qwen2.5", "params": "3B", "quant": "Q5_K_M", "binary": "official",
        "kv": ["q8_0","q8_0"], "extra": "--reasoning auto"},
}

PROMPT = ("You are an expert Python developer. Write a function that implements "
    "a binary search tree with insert, delete, and search operations. "
    "Include type hints, docstrings, and a simple test. Be thorough but concise.")

CTX_SIZES = [4096, 16384, 32768, 65536, 131072]

def init_db():
    db = sqlite3.connect(DB_PATH)
    db.execute("""CREATE TABLE IF NOT EXISTS benchmarks (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT DEFAULT (datetime('now')),
        model_name TEXT, architecture TEXT, params TEXT, quant TEXT, binary TEXT,
        ctx_size INTEGER, kv_k TEXT, kv_v TEXT, extra_args TEXT,
        prompt_tok_s REAL, gen_tok_s REAL, vram_mib INTEGER,
        prompt_tokens INTEGER, completion_tokens INTEGER, content_preview TEXT,
        UNIQUE(model_name, ctx_size, binary))""")
    db.commit()
    return db

def get_vram():
    try:
        out = subprocess.check_output(["nvidia-smi","--query-gpu=memory.used",
            "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL).decode().strip()
        return int(out.split(",")[0].strip())
    except: return -1

def wait_health(port, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return True
        except: time.sleep(2)
    return False

def kill_proc(proc):
    if proc.poll() is None:
        proc.terminate()
        try: proc.wait(timeout=10)
        except: proc.kill(); proc.wait()

def run_bench(port):
    data = json.dumps({"messages": [{"role":"user","content":PROMPT}],
        "max_tokens": 200, "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=data, headers={"Content-Type":"application/json"})
    resp = urllib.request.urlopen(req, timeout=180)
    result = json.loads(resp.read())
    t = result.get("timings", {})
    u = result.get("usage", {})
    c = result.get("choices",[{}])[0].get("message",{}).get("content","")
    return {"prompt_tok_s": t.get("prompt_per_second",0), "gen_tok_s": t.get("predicted_per_second",0),
            "prompt_tokens": u.get("prompt_tokens",0), "completion_tokens": u.get("completion_tokens",0),
            "content_preview": c[:80].replace("\n"," ")}

def benchmark_model(db, name, ctx_sizes):
    cfg = MODELS[name]
    if not Path(cfg["file"]).exists():
        print(f"  [SKIP] {name}: file not found"); return
    binary = BEE_BIN if cfg["binary"] == "bee" else OFFICIAL_BIN
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None); env["CUDA_VISIBLE_DEVICES"] = "0"
    if cfg["binary"] != "bee": env["LD_LIBRARY_PATH"] = OFFICIAL_LIB
    for ctx in ctx_sizes:
        existing = db.execute("SELECT gen_tok_s FROM benchmarks WHERE model_name=? AND ctx_size=? AND binary=?",
            (name, ctx, cfg["binary"])).fetchone()
        if existing:
            print(f"  [{name}] ctx={ctx:>6} cached: {existing[0]:.1f} tok/s"); continue
        print(f"  [{name}] ctx={ctx:>6} starting...", end="", flush=True)
        cmd = [binary, "-m", cfg["file"], "--host","127.0.0.1","--port",str(PORT),
               "-ngl","99","-c",str(ctx),"-t","8","-tb","16","-np","1","-fa","on",
               "--jinja","--no-context-shift",
               "--cache-type-k",cfg["kv"][0],"--cache-type-v",cfg["kv"][1]]
        if cfg["extra"]: cmd += cfg["extra"].split()
        log = open(f"/tmp/bench_{name}_{ctx}.log", "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        if not wait_health(PORT, 120):
            print(f" FAILED"); kill_proc(proc); log.close(); continue
        try: run_bench(PORT)  # warmup
        except: pass
        try: r = run_bench(PORT)
        except Exception as e: print(f" ERROR: {e}"); kill_proc(proc); log.close(); continue
        vram = get_vram(); kill_proc(proc); log.close(); time.sleep(4)
        print(f" gen={r['gen_tok_s']:>6.1f} prompt={r['prompt_tok_s']:>7.1f} vram={vram}MiB")
        db.execute("""INSERT OR REPLACE INTO benchmarks
            (model_name,architecture,params,quant,binary,ctx_size,kv_k,kv_v,extra_args,
             prompt_tok_s,gen_tok_s,vram_mib,prompt_tokens,completion_tokens,content_preview)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (name,cfg["arch"],cfg["params"],cfg["quant"],cfg["binary"],ctx,cfg["kv"][0],cfg["kv"][1],
             cfg["extra"],r["prompt_tok_s"],r["gen_tok_s"],vram,r["prompt_tokens"],
             r["completion_tokens"],r["content_preview"]))
        db.commit()

def report(db):
    rows = db.execute("""SELECT model_name,architecture,params,quant,binary,ctx_size,
        kv_k,gen_tok_s,prompt_tok_s,vram_mib FROM benchmarks ORDER BY params,ctx_size""").fetchall()
    if not rows: print("No benchmark data yet."); return
    print(f"\n{'='*105}")
    print(f"{'Model':<18} {'Arch':<13} {'Params':<8} {'Quant':<8} {'Bin':<5} {'Ctx':>7} {'KV':>5} "
          f"{'Gen t/s':>8} {'Prompt t/s':>10} {'VRAM MB':>8}")
    print(f"{'-'*105}")
    for r in rows:
        print(f"{r[0]:<18} {r[1]:<13} {r[2]:<8} {r[3]:<8} {r[4]:<5} {r[5]:>7} {r[6]:>5} "
              f"{r[7]:>8.1f} {r[8]:>10.1f} {r[9]:>8}")
    print(f"{'='*105}")

def main():
    p = argparse.ArgumentParser(description="Inference speed benchmarks")
    p.add_argument("--models", help="Comma-separated model names")
    p.add_argument("--ctx", help="Comma-separated context sizes")
    p.add_argument("--report", action="store_true", help="Show results table")
    args = p.parse_args()
    db = init_db()
    if args.report: report(db); return
    models = args.models.split(",") if args.models else list(MODELS.keys())
    ctx_sizes = [int(x) for x in args.ctx.split(",")] if args.ctx else CTX_SIZES
    for name in models:
        if name not in MODELS: print(f"  [SKIP] Unknown: {name}"); continue
        print(f"\nBenchmarking: {name}")
        model_ctx = ctx_sizes
        if MODELS[name]["params"] in ("8B-A1B",): model_ctx = [c for c in ctx_sizes if c <= 65536]
        benchmark_model(db, name, model_ctx)
    report(db)

if __name__ == "__main__":
    main()
