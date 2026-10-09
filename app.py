"""Local browser demo, using Python's standard HTTP server.

Run: python app.py, then visit http://127.0.0.1:8000.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

from absa.core import Analyzer, ROOT


def create_handler(analyzer):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, data, content_type="application/json; charset=utf-8"):
            payload = data if isinstance(data, bytes) else json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/":
                self.respond(200, (ROOT / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/metrics":
                self.respond(200, (ROOT / "results" / "metrics.json").read_bytes())
            elif self.path == "/api/health":
                self.respond(200, {"status": "ready", "targets": list(analyzer.bundle["models"])})
            else:
                self.respond(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/api/analyze":
                self.respond(404, {"error": "Not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 100000:
                    raise ValueError("Request size must be within 100 KB.")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("Expected an object containing text.")
                result = analyzer.analyze(payload.get("text"), bool(payload.get("include_unmentioned", False)))
                self.respond(200, result)
            except (ValueError, TypeError, UnicodeDecodeError) as error:
                self.respond(400, {"error": str(error)})

        def log_message(self, *_):
            pass
    return Handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        analyzer = Analyzer()
    except FileNotFoundError:
        parser.exit(1, "Model missing. Run python train.py first.\n")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), create_handler(analyzer))
    print(f"Hotel review demo ready at http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
