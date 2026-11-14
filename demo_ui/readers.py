"""The fine-tuned message reader for the demo: same model, same prompts, same answers as the evaluation.

Lookup order for one message:
  1. the evaluation's own read cache (.cache/reader, written by aquamind/reader/local_reader.py when the reader server answered), then the
     demo's cache (.cache/demo_ui/reader): every benchmark message was read during the evaluation, so replays need no model at all;
  2. the reader server started by `zsh finetune/serve_reader.sh` (127.0.0.1:8471), if it is running;
  3. the merged fine-tuned model loaded in this process with mlx_lm (Apple Silicon; set AQUAMIND_DEMO_MLX=0 to forbid it);
  4. the rule reader (regular expressions), clearly labelled as a fallback.
Every answer goes through the same schema repair as in the evaluation (aquamind/reader/schema.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.request

from .config import CACHE

_mlx_lock = threading.Lock()
_mlx = {"model": None, "tok": None, "error": None, "loading": False}


def _server_url() -> str | None:
    """Returns: URL of the fine-tuned reader server if it answers, else None."""
    from aquamind.reader.local_reader import DEFAULT_URL
    url = os.environ.get("AGENTIX_READER_URL") or DEFAULT_URL
    try:
        urllib.request.urlopen(f"{url}/v1/models", timeout=1.5).read()
        return url
    except OSError:
        return None


def mlx_allowed() -> bool:
    """Returns: True if the fine-tuned model may be loaded in-process (MLX installed, model present, AQUAMIND_DEMO_MLX != '0')."""
    if os.environ.get("AQUAMIND_DEMO_MLX", "1") == "0":
        return False
    from aquamind.reader.local_reader import MERGED
    try:
        import mlx_lm  # noqa: F401
    except Exception:
        return False
    return os.path.exists(f"{MERGED}/config.json")


def status() -> dict:
    """Returns: reader status for the UI {'server', 'mlx_allowed', 'mlx_loaded', 'mlx_loading', 'mlx_error'}."""
    return {"server": _server_url(), "mlx_allowed": mlx_allowed(), "mlx_loaded": _mlx["model"] is not None,
            "mlx_loading": _mlx["loading"], "mlx_error": _mlx["error"]}


class DemoReader:
    """reader-2.0 / reply-2.0 reads with provenance. `prefer`: 'finetuned' (default) or 'rules'."""

    def __init__(self, places=None, prefer: str = "finetuned", allow_model: bool = True):
        """Reader for the demo: cached answers first, then the reader server, then MLX in-process, then the rules.

        Args: places = Places (for the rule fallback); prefer = 'finetuned' | 'rules'; allow_model = may load the model.
        """
        from aquamind.reader.local_reader import CACHE_DIR, LocalReader
        from aquamind.reader.prompts import FT_READ_SYSTEM, FT_REPLY_SYSTEM
        self.prefer, self.allow_model = prefer, allow_model
        self.read_sys, self.reply_sys = FT_READ_SYSTEM, FT_REPLY_SYSTEM
        self.eval_cache = CACHE_DIR
        self.demo_cache = str(CACHE / "reader")
        # the key the evaluation used: model identity + tuned + max tokens + '' (the reader server answered, ADR in local_reader.py)
        self.mid = "|".join([LocalReader._model_files(), "tuned=True", "max=320", ""])
        self.places = places
        self._rules = None
        self._url = None
        self._url_checked = 0.0
        self.provenance: dict[str, str] = {}          # report id -> where the read came from
        self.stats = {"cache": 0, "server": 0, "mlx": 0, "rules": 0}
        self.name = "finetuned_local" if prefer == "finetuned" else "rules_b0"

    # ------------------------------------------------------------------ plumbing
    def rules(self):
        """Returns: the rule-based reader (created on first use)."""
        if self._rules is None:
            from aquamind.reader.rules import RuleReader
            if self.places is None:
                from .world import town
                self.places = town().places
            self._rules = RuleReader(self.places)
        return self._rules

    def _key(self, system: str, user: str) -> str:
        """Args: system, user = prompts.
        Returns: cache key (same as the evaluation's reader cache).
        """
        return hashlib.sha256(json.dumps([self.mid, system, user], ensure_ascii=False).encode()).hexdigest()

    def _cache_get(self, key: str):
        """Look in the evaluation cache, then the demo cache.

        Args: key = cache key.
        Returns: cached answer or None.
        """
        for d in (self.eval_cache, self.demo_cache):
            p = os.path.join(d, key[:2], key + ".json")
            if os.path.exists(p):
                try:
                    with open(p) as f:
                        return json.load(f)
                except ValueError:
                    pass
        return None

    def _cache_put(self, key: str, out) -> None:
        """Save an answer in the demo cache (atomic write).

        Args: key = cache key; out = answer.
        """
        p = os.path.join(self.demo_cache, key[:2], key + ".json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, ensure_ascii=False)
        os.replace(tmp, p)

    def _server(self):
        """Returns: reader server URL (re-checked at most every 20 s) or None."""
        if time.time() - self._url_checked > 20:
            self._url, self._url_checked = _server_url(), time.time()
        return self._url

    def _gen_server(self, url, system, user):
        """Ask the reader server.

        Args: url = server URL; system, user = prompts.
        Returns: parsed JSON dict or None.
        """
        body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "max_tokens": 320, "temperature": 0.0}
        req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            out = json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"]
            return json.loads(out[out.find("{"): out.rfind("}") + 1])
        except (OSError, ValueError, KeyError):
            return None

    def _gen_mlx(self, system, user):
        """Ask the fine-tuned model in-process with MLX (loads it on first use).

        Args: system, user = prompts.
        Returns: parsed JSON dict or None.
        """
        if not (self.allow_model and mlx_allowed()):
            return None
        with _mlx_lock:
            if _mlx["model"] is None and _mlx["error"] is None:
                _mlx["loading"] = True
                try:
                    from mlx_lm import load
                    from aquamind.reader.local_reader import MERGED
                    _mlx["model"], _mlx["tok"] = load(MERGED)
                except Exception as e:                      # noqa: BLE001 - any failure means: use the next fallback
                    _mlx["error"] = repr(e)[:200]
                finally:
                    _mlx["loading"] = False
            if _mlx["model"] is None:
                return None
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler
            tok = _mlx["tok"]
            prompt = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                             tokenize=False, add_generation_prompt=True)
            out = generate(_mlx["model"], tok, prompt=prompt, max_tokens=320, sampler=make_sampler(temp=0.0))
        try:
            return json.loads(out[out.find("{"): out.rfind("}") + 1])
        except ValueError:
            return None

    def gen(self, system: str, user: str) -> tuple[object, str]:
        """-> (raw model answer or None, provenance)."""
        key = self._key(system, user)
        hit = self._cache_get(key)
        if hit is not None:
            self.stats["cache"] += 1
            return hit, "cache"
        if not self.allow_model:
            return None, "none"
        url = self._server()
        if url:
            out = self._gen_server(url, system, user)
            if out is not None:
                self._cache_put(key, out)
                self.stats["server"] += 1
                return out, "server"
        out = self._gen_mlx(system, user)
        if out is not None:
            self._cache_put(key, out)
            self.stats["mlx"] += 1
            return out, "mlx"
        return None, "none"

    # ------------------------------------------------------------------ the reader interface the toolbox uses
    def read_one(self, report: dict) -> tuple[dict, str]:
        """Read one message and say where the answer came from.

        Args: report = message dict.
        Returns: (read dict, provenance: 'cache' | 'server' | 'mlx' | 'rules').
        """
        from aquamind.reader.schema import repair_read
        if self.prefer == "rules":
            self.stats["rules"] += 1
            return self.rules().read(report), "rules"
        user = json.dumps({"channel": report.get("channel"), "text": report["text"]}, ensure_ascii=False)
        raw, prov = self.gen(self.read_sys, user)
        if raw is None:
            self.stats["rules"] += 1
            return self.rules().read(report), "rules-fallback"
        return repair_read(raw), prov

    def read(self, report: dict) -> dict:
        """Args: report = message dict.
        Returns: read dict (provenance stored in self.provenance).
        """
        rd, prov = self.read_one(report)
        self.provenance[report.get("report_id", "?")] = prov
        return rd

    def read_many(self, reports: list[dict]) -> dict[str, dict]:
        """Args: reports = message dicts.
        Returns: {report_id: read dict}.
        """
        return {r["report_id"]: self.read(r) for r in reports}

    def read_replies(self, items: list[tuple[str, str | None]]) -> list[dict]:
        """Args: items = [(question, answer or None)].
        Returns: one reply dict per item.
        """
        from aquamind.reader.schema import EMPTY_REPLY, repair_reply
        if self.prefer == "rules":
            return self.rules().read_replies(items)
        out = []
        for q, a in items:
            if not a:
                out.append(dict(EMPTY_REPLY))
                continue
            raw, prov = self.gen(self.reply_sys, json.dumps({"question": q, "answer": a}, ensure_ascii=False))
            out.append(repair_reply(raw) if raw is not None else self.rules().read_replies([(q, a)])[0])
        return out
