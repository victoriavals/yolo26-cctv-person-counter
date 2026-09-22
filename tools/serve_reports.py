"""Serve a folder over HTTP with Range support, so videos can be seeked.

VSCode's built-in video preview cannot play files in a Remote session -- the
file has to cross the remote channel and the preview fails regardless of codec.
Serving over HTTP sidesteps it: VSCode forwards the port automatically, so the
URL opens in Simple Browser or any local browser.

Python's http.server answers every request with 200 and the whole file, so a
browser cannot seek. This adds 206 Partial Content.

    python tools/serve_reports.py --dir reports/tracking --port 8910
"""
import argparse
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        rng = self.headers.get("Range")
        if not rng:
            return super().send_head()
        path = self.translate_path(self.path)
        if os.path.isdir(path) or not os.path.exists(path):
            return super().send_head()

        size = os.path.getsize(path)
        m = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
        if not m:
            return super().send_head()
        first, last = m.group(1), m.group(2)
        if first == "":                      # suffix range: last N bytes
            start, end = max(0, size - int(last)), size - 1
        else:
            start = int(first)
            end = int(last) if last else size - 1
        if start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        end = min(end, size - 1)

        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        if not hasattr(self, "_remaining"):
            return super().copyfile(source, outputfile)
        left = self._remaining
        del self._remaining
        while left > 0:
            chunk = source.read(min(64 * 1024, left))
            if not chunk:
                break
            outputfile.write(chunk)
            left -= len(chunk)

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass                                  # keep the console quiet


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="reports/tracking")
    ap.add_argument("--port", type=int, default=8910)
    ap.add_argument("--bind", default="127.0.0.1")
    a = ap.parse_args()

    d = a.dir if os.path.isabs(a.dir) else os.path.join(root, a.dir)
    if not os.path.isdir(d):
        raise SystemExit(f"bukan direktori: {d}")
    handler = partial(RangeHandler, directory=d)
    print(f"melayani {d}")
    print(f"  http://{a.bind}:{a.port}/            (daftar file)")
    print(f"  http://{a.bind}:{a.port}/player.html (pemutar, kalau ada)")
    print("Ctrl-C untuk berhenti.")
    ThreadingHTTPServer((a.bind, a.port), handler).serve_forever()


if __name__ == "__main__":
    main()
