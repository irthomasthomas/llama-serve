#!/usr/bin/env python3
"""Unit tests for router/llama-guard.py.

Spins up a fake OpenAI-compatible backend (http.server in a thread) whose
listed models can be toggled at runtime, plus guard instances pointing at it.

Run:  python3 tests/test_guard.py -v
"""

import http.client
import importlib.util
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
GUARD_PATH = os.path.join(HERE, "..", "router", "llama-guard.py")

spec = importlib.util.spec_from_file_location("llama_guard", GUARD_PATH)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

RESIDENT = ["lfm2.5-8b-a1b", "lfm2.5-1.2b-instruct"]
SWAP_MODEL = "vibethinker-3b"


# --------------------------------------------------------------------------
# Fake backend
# --------------------------------------------------------------------------
class FakeBackendHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": m, "object": "model"} for m in self.server.models]})
        elif self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        payload = json.loads(raw or b"{}")
        self.server.posts.append((self.path, payload))
        if self.path == "/v1/chat/completions":
            self._json(200, {
                "id": "chatcmpl-test", "object": "chat.completion",
                "model": payload.get("model"),
                "choices": [{"index": 0, "message": {
                    "role": "assistant", "content": "ok"},
                    "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2}})
        elif self.path == "/v1/completions":
            self._json(200, {"id": "cmpl-test", "object": "text_completion",
                             "model": payload.get("model"),
                             "choices": [{"index": 0, "text": "ok",
                                          "finish_reason": "stop"}]})
        elif self.path == "/v1/embeddings":
            self._json(200, {"object": "list",
                             "data": [{"object": "embedding", "index": 0,
                                       "embedding": [0.1, 0.2]}]})
        elif self.path == "/v1/rerank":
            self._json(200, {"results": [{"index": 0, "relevance_score": 0.9}]})
        else:
            self._json(404, {"error": "not found"})


class FakeBackend(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, models):
        super().__init__(("127.0.0.1", 0), FakeBackendHandler)
        self.models = list(models)   # toggle at runtime: server.models[:] = [...]
        self.posts = []              # every POST the backend actually received


def serve(srv):
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


def request(port, method, path, payload=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    body = json.dumps(payload).encode() if payload is not None else None
    hdrs = dict(headers or {})
    if body is not None:
        hdrs.setdefault("Content-Type", "application/json")
    conn.request(method, path, body=body, headers=hdrs)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    try:
        return resp.status, json.loads(raw)
    except json.JSONDecodeError:
        return resp.status, raw


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
class GuardTestCase(unittest.TestCase):
    dry_run = False

    def setUp(self):
        self.backend = serve(FakeBackend(RESIDENT))
        bport = self.backend.server_address[1]
        self.guard = serve(guard.GuardServer(
            ("127.0.0.1", 0), "127.0.0.1", bport, dry_run=self.dry_run))
        self.port = self.guard.server_address[1]

    def tearDown(self):
        self.guard.shutdown()
        self.guard.server_close()
        self.backend.shutdown()
        self.backend.server_close()

    def chat(self, model, headers=None):
        return request(self.port, "POST", "/v1/chat/completions",
                       {"model": model,
                        "messages": [{"role": "user", "content": "hi"}]},
                       headers=headers)


# --------------------------------------------------------------------------
# Tests: normal (forwarding) mode
# --------------------------------------------------------------------------
class TestGuard(GuardTestCase):
    dry_run = False

    def test_resident_model_200(self):
        status, body = self.chat(RESIDENT[0])
        self.assertEqual(status, 200)
        self.assertEqual(body["model"], RESIDENT[0])
        self.assertEqual(body["choices"][0]["message"]["content"], "ok")
        # request actually reached the backend
        self.assertEqual([p for p, _ in self.backend.posts],
                         ["/v1/chat/completions"])

    def test_non_resident_model_409(self):
        status, body = self.chat(SWAP_MODEL)
        self.assertEqual(status, 409)
        err = body["error"]
        self.assertEqual(err["type"], "model_swap_guard")
        self.assertEqual(err["message"],
                         f"{SWAP_MODEL} requires evicting a resident model "
                         "(~3-5s reload)")
        self.assertEqual(err["alternatives"], RESIDENT)
        self.assertEqual(err["override"],
                         "resend with header X-Allow-Swap: 1 or choose an "
                         "alternative model")
        # blocked: backend never saw the chat request
        self.assertEqual(self.backend.posts, [])

    def test_non_resident_completions_also_guarded(self):
        status, body = request(self.port, "POST", "/v1/completions",
                               {"model": SWAP_MODEL, "prompt": "hi"})
        self.assertEqual(status, 409)
        self.assertEqual(body["error"]["type"], "model_swap_guard")
        self.assertEqual(self.backend.posts, [])

    def test_override_header_200(self):
        status, body = self.chat(SWAP_MODEL, headers={"X-Allow-Swap": "1"})
        self.assertEqual(status, 200)
        self.assertEqual(body["model"], SWAP_MODEL)
        self.assertEqual([p for p, _ in self.backend.posts],
                         ["/v1/chat/completions"])

    def test_toggle_models_flips_decision(self):
        # backend evicts the 8b and loads the swap model instead
        self.backend.models[:] = [SWAP_MODEL, RESIDENT[1]]
        status, _ = self.chat(SWAP_MODEL)
        self.assertEqual(status, 200)
        self.backend.posts.clear()
        # ...and the previously resident model now 409s
        status, body = self.chat(RESIDENT[0])
        self.assertEqual(status, 409)
        self.assertEqual(body["error"]["alternatives"],
                         [SWAP_MODEL, RESIDENT[1]])
        self.assertEqual(self.backend.posts, [])

    def test_health_passthrough(self):
        status, body = request(self.port, "GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "ok"})

    def test_models_passthrough(self):
        status, body = request(self.port, "GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual([m["id"] for m in body["data"]], RESIDENT)

    def test_embeddings_unguarded_even_for_non_resident(self):
        status, body = request(self.port, "POST", "/v1/embeddings",
                               {"model": SWAP_MODEL, "input": "hello"})
        self.assertEqual(status, 200)
        self.assertIn("data", body)
        self.assertEqual([p for p, _ in self.backend.posts],
                         ["/v1/embeddings"])

    def test_rerank_unguarded(self):
        status, body = request(self.port, "POST", "/v1/rerank",
                               {"model": "lfm2-colbert-350m",
                                "query": "q", "documents": ["d"]})
        self.assertEqual(status, 200)
        self.assertIn("results", body)


# --------------------------------------------------------------------------
# Tests: --dry-run mode (decisions logged/returned, nothing forwarded)
# --------------------------------------------------------------------------
class TestGuardDryRun(GuardTestCase):
    dry_run = True

    def test_dry_run_resident_no_side_effects(self):
        status, body = self.chat(RESIDENT[0])
        self.assertEqual(status, 200)
        self.assertTrue(body["dry_run"])
        self.assertEqual(body["decision"], "allow")
        self.assertEqual(body["model"], RESIDENT[0])
        self.assertEqual(self.backend.posts, [])   # nothing forwarded

    def test_dry_run_non_resident_409_shape_no_side_effects(self):
        status, body = self.chat(SWAP_MODEL)
        self.assertEqual(status, 409)
        self.assertTrue(body["dry_run"])
        self.assertEqual(body["error"]["type"], "model_swap_guard")
        self.assertEqual(body["error"]["alternatives"], RESIDENT)
        self.assertEqual(self.backend.posts, [])   # nothing forwarded

    def test_dry_run_override_still_not_forwarded(self):
        status, body = self.chat(SWAP_MODEL, headers={"X-Allow-Swap": "1"})
        self.assertEqual(status, 200)
        self.assertTrue(body["dry_run"])
        self.assertEqual(body["decision"], "allow")
        self.assertEqual(self.backend.posts, [])   # override logged, not sent

    def test_dry_run_passthrough_still_works(self):
        # unguarded endpoints are not decisions; they still forward
        status, body = request(self.port, "GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
