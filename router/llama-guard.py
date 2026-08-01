#!/usr/bin/env python3
"""
llama-guard.py — model-eviction guard proxy for a llama-server router.

Listens on :8091 (default) and forwards OpenAI-compatible requests to a
backend llama-server router on :8080 (default). Before forwarding
POST /v1/chat/completions or /v1/completions it asks the backend which models
are currently resident (GET /v1/models). If the requested model is NOT
resident, loading it would evict a resident model (~3-5s mmap page-cache
reload), so the guard answers HTTP 409 instead of silently stalling the
request. Clients may force the swap with header `X-Allow-Swap: 1`.

Unguarded (always forwarded): /v1/models, /health, /v1/embeddings,
/v1/rerank, and any other path.

--dry-run: compute and log the decision and report it to the caller, but
never forward guarded requests to the backend (no side effects).

Stdlib only. Structured JSON-lines logging to stderr.
"""

import argparse
import http.client
import json
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GUARDED_PATHS = ("/v1/chat/completions", "/v1/completions")
OVERRIDE_HEADER = "X-Allow-Swap"
HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
})


def log(event, **fields):
    rec = {"ts": round(time.time(), 3), "event": event, **fields}
    sys.stderr.write(json.dumps(rec) + "\n")
    sys.stderr.flush()


def swap_guard_error(model, alternatives):
    return {"error": {
        "type": "model_swap_guard",
        "message": f"{model} requires evicting a resident model (~3-5s reload)",
        "alternatives": list(alternatives),
        "override": "resend with header X-Allow-Swap: 1 or choose an alternative model",
    }}


class GuardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, backend_host, backend_port, dry_run=False, timeout=30.0):
        super().__init__(addr, GuardHandler)
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.dry_run = dry_run
        self.timeout = timeout

    def loaded_models(self):
        """GET /v1/models from the backend -> list of model ids, or None on error."""
        conn = http.client.HTTPConnection(self.backend_host, self.backend_port,
                                          timeout=self.timeout)
        try:
            conn.request("GET", "/v1/models", headers={"Accept": "application/json"})
            resp = conn.getresponse()
            body = resp.read()
            if resp.status != 200:
                log("models_query_error", status=resp.status)
                return None
            data = json.loads(body)
            return [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
        except (OSError, json.JSONDecodeError) as e:
            log("models_query_error", error=str(e))
            return None
        finally:
            conn.close()

    def forward(self, method, path, headers, body):
        """Forward a request to the backend; return (status, headers, body)."""
        conn = http.client.HTTPConnection(self.backend_host, self.backend_port,
                                          timeout=self.timeout)
        try:
            fwd = {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}
            conn.request(method, path, body=body, headers=fwd)
            resp = conn.getresponse()
            resp_body = resp.read()
            resp_headers = [(k, v) for k, v in resp.getheaders()
                            if k.lower() not in HOP_BY_HOP]
            return resp.status, resp_headers, resp_body
        finally:
            conn.close()


class GuardHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "llama-guard/1.0"

    def log_message(self, fmt, *args):
        log("access", client=self.client_address[0], msg=fmt % args)

    # ---- response helpers ----------------------------------------------
    def _send_json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_raw(self, status, headers, body):
        self.send_response(status)
        has_len = any(k.lower() == "content-length" for k, _ in headers)
        for k, v in headers:
            self.send_header(k, v)
        if not has_len:
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    # ---- routing --------------------------------------------------------
    def _handle(self, method):
        path = self.path
        body = self._read_body() if method in ("POST", "PUT", "PATCH", "DELETE") else None
        guarded = method == "POST" and path.split("?", 1)[0] in GUARDED_PATHS
        if guarded:
            self._guarded(method, path, body)
        else:
            self._passthrough(method, path, body)

    def _passthrough(self, method, path, body):
        try:
            status, headers, resp_body = self.server.forward(method, path, self.headers, body)
        except OSError as e:
            log("backend_unreachable", path=path, error=str(e))
            return self._send_json(502, {"error": {"type": "backend_unreachable",
                                                   "message": str(e)}})
        log("passthrough", method=method, path=path, status=status)
        self._send_raw(status, headers, resp_body)

    def _guarded(self, method, path, body):
        try:
            model = json.loads(body or b"{}").get("model")
        except json.JSONDecodeError:
            return self._send_json(400, {"error": {"type": "invalid_json",
                                                   "message": "request body is not valid JSON"}})

        override = (self.headers.get(OVERRIDE_HEADER, "") or "").strip() == "1"

        # No model field -> let the backend produce its own error.
        if not model:
            log("decision", decision="forward", reason="no_model_field",
                dry_run=self.server.dry_run)
            if self.server.dry_run:
                return self._send_json(200, {"dry_run": True, "decision": "forward",
                                             "reason": "no_model_field"})
            return self._forward_guarded(method, path, body, model)

        loaded = self.server.loaded_models()
        if loaded is None:
            # Fail-open: backend may be mid-swap; forward rather than block.
            log("decision", decision="fail_open", reason="models_query_failed",
                model=model, dry_run=self.server.dry_run)
            if self.server.dry_run:
                return self._send_json(200, {"dry_run": True, "decision": "fail_open",
                                             "model": model})
            return self._forward_guarded(method, path, body, model)

        resident = model in loaded
        decision = "allow" if (resident or override) else "block"
        log("decision", decision=decision, model=model, resident=resident,
            override=override, loaded=loaded, dry_run=self.server.dry_run)

        if self.server.dry_run:
            if decision == "allow":
                return self._send_json(200, {"dry_run": True, "decision": "allow",
                                             "model": model, "loaded": loaded})
            return self._send_json(409, {"dry_run": True, **swap_guard_error(model, loaded)})

        if decision == "block":
            return self._send_json(409, swap_guard_error(model, loaded))
        self._forward_guarded(method, path, body, model)

    def _forward_guarded(self, method, path, body, model):
        try:
            status, headers, resp_body = self.server.forward(method, path, self.headers, body)
        except OSError as e:
            log("backend_unreachable", path=path, error=str(e))
            return self._send_json(502, {"error": {"type": "backend_unreachable",
                                                   "message": str(e)}})
        log("forwarded", method=method, path=path, model=model, status=status)
        self._send_raw(status, headers, resp_body)

    def do_GET(self):    self._handle("GET")
    def do_POST(self):   self._handle("POST")
    def do_PUT(self):    self._handle("PUT")
    def do_DELETE(self): self._handle("DELETE")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Model-eviction guard proxy for a llama-server router")
    p.add_argument("--listen-host", default="127.0.0.1")
    p.add_argument("--listen-port", type=int, default=8091)
    p.add_argument("--backend-host", default="127.0.0.1")
    p.add_argument("--backend-port", type=int, default=8080)
    p.add_argument("--dry-run", action="store_true",
                   help="log decisions without forwarding guarded requests")
    p.add_argument("--timeout", type=float, default=30.0)
    args = p.parse_args(argv)

    srv = GuardServer((args.listen_host, args.listen_port),
                      args.backend_host, args.backend_port,
                      dry_run=args.dry_run, timeout=args.timeout)
    log("listening", host=args.listen_host, port=args.listen_port,
        backend=f"{args.backend_host}:{args.backend_port}", dry_run=args.dry_run)
    # graceful SIGTERM: shutdown must run in a thread (serve_forever holds
    # the main thread; calling shutdown() from the same thread would deadlock)
    def _sigterm(_sig, _frame):
        log("shutdown", reason="SIGTERM")
        threading.Thread(target=srv.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, _sigterm)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
