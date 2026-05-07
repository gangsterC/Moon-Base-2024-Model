#!/usr/bin/env python3
"""
Tiny server for precomputed TagPro MLP prediction JSON files.

It mimics the overlay server endpoints:
  GET /health
  GET /list
  GET /predict?file=<prediction-json-file>

This lets the existing Tampermonkey overlay work without running inference live.
"""

import argparse
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote


def make_handler(pred_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, obj, code=200):
            data = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self._send_json({"ok": True})

        def do_GET(self):
            try:
                parsed = urlparse(self.path)
                qs = parse_qs(parsed.query)

                if parsed.path == "/health":
                    self._send_json({"ok": True, "service": "tagpro_precomputed_overlay_server"})
                    return

                if parsed.path == "/list":
                    files = sorted(pred_dir.glob("*_mlp_predictions.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                    self._send_json({
                        "ok": True,
                        "files": [
                            {
                                "name": p.name,
                                "path": str(p),
                                "size": p.stat().st_size,
                                "mtime": p.stat().st_mtime,
                            }
                            for p in files
                        ]
                    })
                    return

                if parsed.path == "/predict":
                    name = unquote(qs.get("file", [""])[0])
                    if not name:
                        files = sorted(pred_dir.glob("*_mlp_predictions.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                        if not files:
                            raise FileNotFoundError(f"No *_mlp_predictions.json files in {pred_dir}")
                        p = files[0]
                    else:
                        p = Path(name)
                        if not p.exists():
                            p = pred_dir / name
                    if not p.exists():
                        raise FileNotFoundError(f"Prediction JSON not found: {name}")

                    self._send_json(json.loads(p.read_text()))
                    return

                self._send_json({"ok": False, "error": "Unknown endpoint"}, code=404)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, code=500)

        def log_message(self, fmt, *args):
            print("%s - %s" % (self.address_string(), fmt % args))

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8767)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir).expanduser()
    pred_dir.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(pred_dir))
    print(f"Serving precomputed predictions from: {pred_dir}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Chrome usually reaches this from: http://penguin.linux.test:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
