"""Tiny static file server with HTTP Range (206) support -- a throwaway diagnostic
used to prove the remote avatar lag is the live WebRTC path, not the render.

Why this exists: serving an already-rendered avatar mp4 over HTTP/TCP lets the
browser buffer the whole clip ahead, so it plays smoothly whenever the link has
enough *average* throughput. If the remote viewer sees the mp4 smooth but the
live WebRTC avatar lags, the fault is the real-time transport (fit-the-stream
fix), not the system. Chromium needs 206 Range replies to start mp4 playback
reliably, which stdlib SimpleHTTPRequestHandler does not provide -- hence this.

Usage:
    python -m scripts.serve_clip --dir output/_serve --host 127.0.0.1 --port 8090
Expose to the remote viewer over the existing Tailscale host on a side path:
    tailscale serve --bg --set-path /watch http://127.0.0.1:8090
    -> https://<machine>.<tailnet>.ts.net/watch/
"""
from __future__ import annotations

import argparse
import os
import re
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class RangeRequestHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler + single-range (206 Partial Content) support."""

    def send_head(self):  # noqa: C901 - mirrors stdlib structure
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            # Defer directory handling (index.html / listing) to the base class.
            return super().send_head()
        rng = self.headers.get("Range")
        if not rng:
            return super().send_head()

        m = _RANGE_RE.fullmatch(rng.strip())
        if not m:
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None

        try:
            fs = os.fstat(f.fileno())
            size = fs.st_size
            start_s, end_s = m.group(1), m.group(2)
            if start_s == "":
                # Suffix range: last N bytes.
                length = int(end_s)
                start = max(0, size - length)
                end = size - 1
            else:
                start = int(start_s)
                end = int(end_s) if end_s else size - 1
            end = min(end, size - 1)
            if start > end or start >= size:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                f.close()
                return None

            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Last-Modified", self.date_time_string(int(fs.st_mtime)))
            self.end_headers()
            f.seek(start)
            self._range = (start, end)  # consumed by copyfile override
            return f
        except Exception:
            f.close()
            raise

    def copyfile(self, source, outputfile):
        rng = getattr(self, "_range", None)
        if rng is None:
            return super().copyfile(source, outputfile)
        self._range = None
        start, end = rng
        remaining = end - start + 1
        chunk = 64 * 1024
        while remaining > 0:
            buf = source.read(min(chunk, remaining))
            if not buf:
                break
            outputfile.write(buf)
            remaining -= len(buf)


def main() -> None:
    ap = argparse.ArgumentParser(description="Range-capable static server (diagnostic).")
    ap.add_argument("--dir", default="output/_serve", help="directory to serve")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()

    directory = os.path.abspath(args.dir)
    handler = partial(RangeRequestHandler, directory=directory)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {directory} at http://{args.host}:{args.port}/ (Range-capable)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
