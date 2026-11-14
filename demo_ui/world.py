"""Shared, lazily loaded AquaMind objects (one per server process).

The evaluation World (town, truth simulator, documents, SOP rule book) loads in about 2 s; the hypothesis-mode leak search keeps a physics memo
that makes later runs faster. Both are built once, behind a lock, and shared by every request.
"""
from __future__ import annotations

import threading

from .config import BENCH

_lock = threading.RLock()
_world = None
_search = None

# exactly the search settings of the AquaMind crew (aquamind/methods.py: H = {"hypothesis_mode": True, "reply_confirms": False})
CREW_SEARCH = {"hypothesis_mode": True, "reply_confirms": False}
# exactly the toolbox settings of the AquaMind crew (aquamind/methods.py: use_planner=False, min_checks_close=3)
CREW_TOOLBOX = {"use_planner": False, "min_checks_close": 3}


def world():
    """Returns: the shared World for the benchmark (created once, thread-safe)."""
    global _world
    with _lock:
        if _world is None:
            from aquamind.eval.runner import World
            _world = World(BENCH)
        return _world


def search():
    """The crew's LeakSearch (shared: its physics memo is a pure compute cache)."""
    global _search
    with _lock:
        if _search is None:
            _search = world().search(**CREW_SEARCH)
        return _search


def town():
    """Returns: the shared Town."""
    return world().town


# one heavy run (live investigation, recording, attack sandbox) at a time: the physics tool is not thread-safe
run_lock = threading.Lock()
