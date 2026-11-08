"""End-to-end test of the LLM path against a fake OpenAI-compatible server.
Checks: request format, JSON extraction, cut-ID validation, hallucination rejection."""
import json, os, re, subprocess, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body["model"] == "exhausted-model":  # simulate a free-tier daily cap
            err = json.dumps({"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                        "message": "quota GenerateRequestsPerDayPerProjectPerModel-FreeTier"}}).encode()
            self.send_response(429); self.end_headers(); self.wfile.write(err); return
        prompt = body["messages"][1]["content"]
        k1 = re.search(r"K1: (C\d+) -> (C\d+)", prompt)
        reply = {"promos": [
            {"id": "K1", "start_cut": k1.group(1), "end_cut": k1.group(2), "hook": "mock hook", "cliffhanger": "mock cliff",
             "why": "mock why", "hook_score": 8, "cliff_score": 7, "standalone_score": 9, "overall": 8, "issues": ""},
            {"id": "K2", "start_cut": "C99", "end_cut": "C98", "hook": "invented", "overall": 10},  # must be rejected
        ]}
        out = {"choices": [{"message": {"content": "Sure! ```json\n" + json.dumps(reply) + "\n```"}}],
               "usage": {"prompt_tokens": 1000, "completion_tokens": 200}}
        data = json.dumps(out).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(data)

srv = HTTPServer(("127.0.0.1", 8765), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
show = sys.argv[1]
r = subprocess.run([sys.executable, "run.py", show, "--provider", "custom", "--base-url", "http://127.0.0.1:8765/v1",
                    "--api-key-env", "", "--model", "exhausted-model,mock", "--episodes", "1-3", "--min-interval", "0",
                    "--out", "/tmp/promo_mock_out", "--cache-dir", tempfile.mkdtemp(), "--no-stitch"], capture_output=True, text=True,
                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
print(r.stdout[-900:], r.stderr[-300:])
d = json.load(open("/tmp/promo_mock_out/promos.json"))
judged = [p for p in d["all_promos"] if p["judged_by"] == "llm(mock)+rules"]
others = [p for p in d["all_promos"] if p["judged_by"] != "llm(mock)+rules"]
assert judged, "LLM verdicts/fallback not applied"
assert all(any("rules-only backup" in f for f in p["flags"]) for p in others), "unjudged clips must be flagged backups"
assert max(p["score"] for p in others) < max(p["score"] for p in judged), "backups must rank below judged clips"
assert "switching to mock" in r.stderr, "quota fallback not triggered"
print("OK: daily-quota error switched model chain exhausted-model -> mock")
assert any("unknown cut id C99" in l for l in d["meta"]["log"]), "hallucinated cut not rejected"
print("OK: LLM path works; invented cut IDs rejected:", [l for l in d["meta"]["log"] if "C99" in l][:1])
srv.shutdown()
