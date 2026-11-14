"""Live investigations: the real crew runs in a background thread and its events stream to the browser (Server-Sent Events).

One run at a time (the physics tool is shared). A finished run is saved as a recording when it is faithful (no missing LLM answer, no crash),
so a case run live before the presentation can be replayed afterwards at any speed.
"""
from __future__ import annotations

import threading
import time
import uuid

from . import instrument, recordings
from .world import run_lock


class LiveRun:
    def __init__(self, case: str, layout: str, online: bool, reader: str):
        self.id = uuid.uuid4().hex[:10]
        self.case, self.layout, self.online, self.reader = case, layout, online, reader
        self.events: list[dict] = []
        self.done = False
        self.saved = None
        self.error = None
        self.started = time.time()
        self.cond = threading.Condition()
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"live-{self.id}")

    def _sink(self, ev: dict) -> None:
        with self.cond:
            self.events.append(ev)
            self.cond.notify_all()

    def _run(self) -> None:
        try:
            with run_lock:
                rec = instrument.run_case(self.case, self.layout, allow_network=self.online, sink=self._sink, reader=self.reader)
            r = rec["result"]
            if not r.get("error") and not r.get("llm_misses"):
                self.saved = recordings.save(rec)
        except Exception as e:             # noqa: BLE001 - report to the browser instead of dying silently
            import traceback
            self.error = f"{e!r}\n{traceback.format_exc()[-1200:]}"
        finally:
            with self.cond:
                self.done = True
                self.cond.notify_all()

    def stream(self, start: int = 0, heartbeat_s: float = 10.0):
        """Yield events from index `start`; yields None as a heartbeat while waiting."""
        i = start
        while True:
            with self.cond:
                while i >= len(self.events) and not self.done:
                    if not self.cond.wait(timeout=heartbeat_s):
                        break
                batch = self.events[i:]
                finished = self.done and i + len(batch) >= len(self.events)
            if batch:
                for ev in batch:
                    yield ev
                i += len(batch)
            else:
                yield None
            if finished:
                return

    def info(self) -> dict:
        return {"id": self.id, "case": self.case, "layout": self.layout, "online": self.online, "reader": self.reader, "events": len(self.events),
                "done": self.done, "saved": self.saved, "error": self.error, "elapsed_s": round(time.time() - self.started, 1)}


RUNS: dict[str, LiveRun] = {}
_lock = threading.Lock()


def start(case: str, layout: str = "spread_k5", online: bool = True, reader: str = "finetuned") -> tuple[LiveRun | None, str | None]:
    with _lock:
        busy = [r for r in RUNS.values() if not r.done]
        if busy:
            return busy[0], "another live investigation is running; it is shown instead"
        run = LiveRun(case, layout, online, reader)
        RUNS[run.id] = run
        run.thread.start()
        return run, None


def get(run_id: str) -> LiveRun | None:
    return RUNS.get(run_id)
