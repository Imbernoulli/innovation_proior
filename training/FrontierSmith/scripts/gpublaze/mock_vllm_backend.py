#!/usr/bin/env python3
"""CPU-only mock vLLM backend for smoke-testing the serve->registry->client
chain on gpublaze without touching any GPU.

Serves the minimal OpenAI surface the eval stack touches:
  GET  /v1/models          -> one model (the TAG)
  GET  /health             -> 200
  GET  /metrics            -> vllm:num_requests_running 0
  POST /v1/chat/completions-> a canned "solution" (C++ code block) so the
                              downstream judge/scoring path gets real work

Usage:  mock_vllm_backend.py --tag TAG --port P [--reply-file f.txt]
The caller writes the registry entry (see eval_smoke_mock.sh).
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CANNED = """<think>mock reasoning</think>
```cpp
#include <bits/stdc++.h>
int main(){ return 0; }
```"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--reply-file", default=None)
    ap.add_argument("--mls-episode", action="store_true",
                    help="scripted MLS agent: reply test() on the first turn, "
                         "submit(n=1) once a tool result is in the transcript -- "
                         "exercises the real conda test runner + scoring chain")
    args = ap.parse_args()
    reply = CANNED if not args.reply_file else open(args.reply_file).read()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/v1/models"):
                self._json({"object": "list", "data": [{"id": args.tag, "object": "model"}]})
            elif self.path == "/health":
                self._json({"ok": True})
            elif self.path == "/metrics":
                body = b"vllm:num_requests_running 0\n"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            if req.get("model") != args.tag:
                self._json({"error": {"message": f"model {req.get('model')!r} not served"}}, 404)
                return
            if args.mls_episode and req.get("tools"):
                had_tool_result = any(m.get("role") == "tool" for m in req.get("messages", []))
                if had_tool_result:
                    fn = {"name": "submit", "arguments": json.dumps({"n": 1})}
                else:
                    fn = {"name": "test", "arguments": "{}"}
                self._json({
                    "id": "mock", "object": "chat.completion", "created": int(time.time()),
                    "model": args.tag,
                    "choices": [{"index": 0, "finish_reason": "tool_calls",
                                 "message": {"role": "assistant", "content": "",
                                             "tool_calls": [{"id": f"call_{int(time.time()*1000)}",
                                                             "type": "function", "function": fn}]}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                })
                return
            nc = req.get("n", 1)
            self._json({
                "id": "mock", "object": "chat.completion", "created": int(time.time()),
                "model": args.tag,
                "choices": [
                    {"index": i, "message": {"role": "assistant", "content": reply},
                     "finish_reason": "stop"} for i in range(nc)
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            })

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    print(f"[mock-vllm] serving tag={args.tag} on 127.0.0.1:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
