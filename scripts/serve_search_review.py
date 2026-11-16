#!/usr/bin/env python3
"""Serve run artifacts and persist blinded expert-search ratings."""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlparse


MAX_BODY_BYTES = 2 * 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def handler_factory(directory: Path, ratings_path: Path):
    write_lock = threading.Lock()

    class ReviewHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def _json_response(self, status, payload):
            body = json.dumps(payload, indent=2, sort_keys=True).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if urlparse(self.path).path != "/api/search-expert-ratings":
                return super().do_GET()
            if not ratings_path.exists():
                return self._json_response(
                    200, {"schemaVersion": 1, "ratings": {}}
                )
            try:
                payload = json.loads(ratings_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                return self._json_response(500, {"error": str(error)})
            return self._json_response(200, payload)

        def do_POST(self):
            if urlparse(self.path).path != "/api/search-expert-ratings":
                return self._json_response(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._json_response(400, {"error": "bad length"})
            if length <= 0 or length > MAX_BODY_BYTES:
                return self._json_response(413, {"error": "invalid body size"})
            try:
                payload = json.loads(self.rfile.read(length))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                return self._json_response(400, {"error": str(error)})
            if not isinstance(payload, dict) or not isinstance(
                payload.get("ratings"), dict
            ):
                return self._json_response(
                    400, {"error": "payload must contain a ratings object"}
                )
            payload["schemaVersion"] = 1
            ratings_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = ratings_path.with_suffix(ratings_path.suffix + ".tmp")
            with write_lock:
                temporary.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n"
                )
                temporary.replace(ratings_path)
            return self._json_response(
                200,
                {
                    "saved": True,
                    "ratings": len(payload["ratings"]),
                },
            )

    return ReviewHandler


def main():
    args = parse_args()
    directory = args.directory.resolve()
    ratings = args.ratings.resolve()
    server = ThreadingHTTPServer(
        (args.bind, args.port), handler_factory(directory, ratings)
    )
    print(
        f"serving {directory} on http://{args.bind}:{args.port}; "
        f"ratings={ratings}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
