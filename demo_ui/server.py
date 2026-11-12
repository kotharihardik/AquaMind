"""A small HTTP server for the demo (Python standard library only: nothing to install, works offline).

Static files from demo_ui/static, JSON endpoints under /api, and Server-Sent Events for live investigations.
"""
from __future__ import annotations

import gzip
import json
import mimetypes
import os
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import geometry, labs, live, logreplay, readers, recordings, results
from .config import BENCH, DEFAULT_LAYOUT, RECORDINGS, STATIC, SUITES, suite_of

_boot = {"ready": False, "error": None, "t": 0.0}


# ---------------------------------------------------------------- data for the views
def catalogue() -> list[dict]:
    idx = json.load(open(f"{BENCH}/index.json"))
    crew = logreplay.outcomes("agentix_crew")
    react = logreplay.outcomes("react_agent")
    recs = {r["case"] for r in recordings.listing() if r.get("layout") == DEFAULT_LAYOUT}
    out = []
    for name, d in idx.items():
        s = d.get("suite") or suite_of(name)
        out.append({"case": name, "split": name.split("-")[0], "suite": s, "suite_name": SUITES.get(s, (s, ""))[0], "has_leak": d.get("has_leak"),
                    "q": d.get("q_mean_lps"), "msgs": d.get("msgs_at_open"), "crew": crew.get(name), "react": react.get(name),
                    "recorded": name in recs})
    return out


def status() -> dict:
    from aquamind.llm.client import _discover_provider_keys, load_dotenv
    load_dotenv()
    return {"ready": _boot["ready"], "boot_error": _boot["error"], "boot_s": _boot["t"], "reader": readers.status(),
            "api_keys": len(_discover_provider_keys("nvidia")), "recordings": len(list(RECORDINGS.glob("*.json"))),
            "live": [r.info() for r in live.RUNS.values() if not r.done]}


def warm() -> None:
    t0 = time.time()
    try:
        from .world import search, world
        world()
        search()
        geometry.payload()
        results.payload()
        logreplay.available("agentix_crew")
        logreplay.available("react_agent")
        # the physics tool's base simulation runs on first use: do it now, not during the first live run
        labs.sop_for(0)
        _boot["ready"] = True
    except Exception as e:                 # noqa: BLE001
        _boot["error"] = f"{e!r}"
        traceback.print_exc()
    _boot["t"] = round(time.time() - t0, 1)


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "AquaMindDemo/1.0"

    def log_message(self, fmt, *args):     # quiet, except errors
        if args and isinstance(args[1], str) and args[1].startswith(("4", "5")):
            super().log_message(fmt, *args)

    # ------------------------------------------------------------ responses
    def _send(self, code: int, body: bytes, ctype: str, cache: bool = False):
        gz = "gzip" in (self.headers.get("Accept-Encoding") or "") and len(body) > 2048 and not ctype.startswith("image/")
        if gz:
            body = gzip.compress(body, 5)
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=300" if cache else "no-store")
        if gz:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode(), "application/json; charset=utf-8")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(min(n, 200_000)) or b"{}")
        except ValueError:
            return {}

    def _static(self, rel: str):
        rel = rel.lstrip("/") or "index.html"
        p = (STATIC / rel).resolve()
        if not str(p).startswith(str(STATIC.resolve())) or not p.is_file():
            return self._json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        if p.suffix == ".js":
            ctype = "text/javascript; charset=utf-8"
        elif p.suffix in (".css", ".html"):
            ctype += "; charset=utf-8"
        self._send(200, p.read_bytes(), ctype)

    # ------------------------------------------------------------ routes
    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(u.query).items()}
        path = u.path
        try:
            if path == "/" or path == "/index.html":
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if path == "/api/status":
                return self._json(status())
            if path == "/api/bootstrap":
                return self._json({"network": geometry.payload(), "cases": catalogue(), "recordings": recordings.listing(), "status": status(),
                                   "suites": {k: {"name": v[0], "blurb": v[1]} for k, v in SUITES.items()}, "examples": labs.examples()})
            if path == "/api/results":
                return self._json(results.payload())
            if path == "/api/recordings":
                return self._json(recordings.listing())
            if path == "/api/recording":
                d = recordings.load(q.get("file", ""))
                return self._json(d) if d else self._json({"error": "no such recording"}, 404)
            if path == "/api/replay":
                d = logreplay.build(q.get("method", "agentix_crew"), q.get("case", ""), q.get("layout", DEFAULT_LAYOUT))
                return self._json(d) if d else self._json({"error": "no log for this case and method"}, 404)
            if path == "/api/h2h":
                c = q.get("case", "")
                return self._json({"crew": logreplay.build("agentix_crew", c), "react": logreplay.build("react_agent", c)})
            if path == "/api/sop":
                return self._json(labs.sop_for(int(q.get("j", 0))))
            if path == "/api/live/stream":
                return self._sse(q.get("id", ""), int(q.get("from", 0) or 0))
            return self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:             # noqa: BLE001
            traceback.print_exc()
            return self._json({"error": repr(e)}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        b = self._body()
        try:
            if u.path == "/api/reader":
                return self._json(labs.read(b.get("text", ""), b.get("channel") or "whatsapp"))
            if u.path == "/api/rag":
                return self._json(labs.rag(b.get("query", ""), int(b.get("k") or 8), b.get("provenance")))
            if u.path == "/api/attack":
                return self._json(labs.attack(b.get("text", ""), b.get("channel") or "whatsapp"))
            if u.path == "/api/live/start":
                run, note = live.start(b.get("case", ""), b.get("layout") or DEFAULT_LAYOUT, bool(b.get("online", True)), b.get("reader") or "finetuned")
                return self._json({**run.info(), "note": note})
            return self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:             # noqa: BLE001
            traceback.print_exc()
            return self._json({"error": repr(e)}, 500)

    def _sse(self, run_id: str, start: int):
        run = live.get(run_id)
        if run is None:
            return self._json({"error": "no such live run"}, 404)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            for ev in run.stream(start):
                if ev is None:
                    self.wfile.write(b": keep-alive\n\n")
                else:
                    self.wfile.write(b"data: " + json.dumps(ev, ensure_ascii=False, default=str).encode() + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"event: done\ndata: " + json.dumps(run.info()).encode() + b"\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    threading.Thread(target=warm, daemon=True, name="warm").start()
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    url = f"http://{host}:{port}/"
    print(f"AquaMind demo running at {url}  (Ctrl+C to stop)", flush=True)
    if open_browser and not os.environ.get("AQUAMIND_DEMO_NO_BROWSER"):
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
