#!/usr/bin/env python3
"""Subprocess-level integration tests for router/llama-guard.py.

Exercises real CLI parsing, the --dry-run flag, the startup "listening" log
line, a guarded 409 response against a stub backend, and graceful shutdown
(SIGTERM). Complements the in-process unittest coverage in test_guard.py.
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUARD = os.path.join(ROOT, "router", "llama-guard.py")

BACKEND_PORT = 18091
GUARD_PORT = 18092


class StubBackend(BaseHTTPRequestHandler):
    """Minimal router stand-in: /v1/models reports one resident model."""
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "lfm2.5-8b-a1b"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class GuardSubprocessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = ThreadingHTTPServer(("127.0.0.1", BACKEND_PORT), StubBackend)
        cls.bt = threading.Thread(target=cls.backend.serve_forever, daemon=True)
        cls.bt.start()

    @classmethod
    def tearDownClass(cls):
        cls.backend.shutdown()
        cls.backend.server_close()

    def _start_guard(self, *extra):
        proc = subprocess.Popen(
            [sys.executable, GUARD,
             "--listen-port", str(GUARD_PORT),
             "--backend-port", str(BACKEND_PORT), *extra],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        # wait for the "listening" log line on stderr
        deadline = time.time() + 10
        listening = False
        self._stderr_lines = []
        while time.time() < deadline:
            line = proc.stderr.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            self._stderr_lines.append(line)
            try:
                if json.loads(line).get("event") == "listening":
                    listening = True
                    break
            except json.JSONDecodeError:
                pass
        self.assertTrue(listening,
                        f"guard never logged 'listening'; stderr={self._stderr_lines!r}")
        return proc

    def _stop_guard(self, proc):
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            self.fail("guard did not shut down on SIGTERM within 5s")

    def _post_chat(self, model, allow_swap=False):
        req = Request(
            f"http://127.0.0.1:{GUARD_PORT}/v1/chat/completions",
            data=json.dumps({"model": model, "messages": []}).encode(),
            headers={"Content-Type": "application/json",
                     **({"X-Allow-Swap": "1"} if allow_swap else {})},
            method="POST",
        )
        try:
            with urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except HTTPError as e:
            return e.code, json.loads(e.read())

    def test_01_startup_listening_and_graceful_shutdown(self):
        proc = self._start_guard()
        self.assertIsNone(proc.poll(), "guard exited prematurely")
        self._stop_guard(proc)
        self.assertEqual(proc.returncode, 0,
                         f"expected clean exit 0, got {proc.returncode}")

    def test_02_guarded_409_for_nonresident_model(self):
        proc = self._start_guard()
        try:
            status, body = self._post_chat("vibethinker-3b")
            self.assertEqual(status, 409)
            self.assertEqual(body["error"]["type"], "model_swap_guard")
            self.assertIn("lfm2.5-8b-a1b", body["error"]["alternatives"])
        finally:
            self._stop_guard(proc)

    def test_03_forwarded_for_resident_model(self):
        proc = self._start_guard()
        try:
            status, body = self._post_chat("lfm2.5-8b-a1b")
            self.assertEqual(status, 200)
            self.assertEqual(body["choices"][0]["message"]["content"], "ok")
        finally:
            self._stop_guard(proc)

    def test_04_dry_run_never_forwards(self):
        proc = self._start_guard("--dry-run")
        try:
            # dry-run: guarded request must NOT reach the backend; the guard
            # reports its decision instead of a real completion.
            status, body = self._post_chat("lfm2.5-8b-a1b")
            self.assertNotEqual(
                body.get("choices", [{}])[0].get("message", {}).get("content"),
                "ok", "dry-run forwarded a guarded request to the backend")
        finally:
            self._stop_guard(proc)

    def test_05_bad_cli_arg_exits_nonzero(self):
        proc = subprocess.run(
            [sys.executable, GUARD, "--listen-port", "not-a-port"],
            capture_output=True, text=True, timeout=15)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("usage", (proc.stderr or proc.stdout).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
