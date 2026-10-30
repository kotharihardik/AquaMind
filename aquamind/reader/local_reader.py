"""The fine-tuned small complaint reader, running locally (proposal: "fast, runs inside the utility").

Base model + LoRA adapter trained by finetune/train_mlx.sh (Apple MLX; finetune/README.md has the GPU/QLoRA recipe).
Every answer is schema-checked and repaired (aquamind/reader/schema.py), so a bad generation can never pass
malformed fields or free text to the agent.
"""
from __future__ import annotations

import hashlib
import json
import os

from .prompts import FT_READ_SYSTEM, FT_REPLY_SYSTEM, READ_SYSTEM, REPLY_SYSTEM
from .schema import EMPTY_REPLY, repair_read, repair_reply

BASE = os.environ.get("AGENTIX_READER_BASE", "mlx-community/Qwen2.5-1.5B-Instruct-4bit")
ADAPTER = os.environ.get("AGENTIX_READER_ADAPTER", "finetune/adapters/reader")
MERGED = "finetune/models/reader_qwen3b_mlx"                   # the Kaggle-trained Qwen2.5-3B QLoRA model, merged and converted (finetune/convert_reader.sh)
CACHE_DIR = os.environ.get("AGENTIX_READER_CACHE", ".cache/reader")   # reads are deterministic (temperature 0): the same model, prompt and text give the same answer
DEFAULT_URL = "http://127.0.0.1:8471"                          # finetune/serve_reader.sh


def _reachable(url: str) -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(f"{url}/v1/models", timeout=3).read()
        return True
    except OSError:
        return False


class LocalReader:
    name = "finetuned_local"

    def __init__(self, base: str = BASE, adapter: str | None = ADAPTER, max_tokens: int = 320):
        self.url = os.environ.get("AGENTIX_READER_URL") or (DEFAULT_URL if _reachable(DEFAULT_URL) else None)
        self.max_tokens = max_tokens
        self.calls = self.errors = self.cached = 0
        if self.url:
            # a server started by finetune/serve_reader.sh serves the fine-tuned model: it learned the short prompts. AGENTIX_READER_TUNED=0 serves the base model on purpose.
            self.tuned = os.environ.get("AGENTIX_READER_TUNED", "1") != "0"
            if not _reachable(self.url):
                raise RuntimeError(f"fine-tuned reader server {self.url} is not answering: start it with `zsh finetune/serve_reader.sh` "
                                   "(a dead reader would silently turn every citizen message into an empty read)")
        else:
            from mlx_lm import load
            if os.path.exists(f"{MERGED}/config.json") and not os.path.exists(f"{adapter}/adapters.safetensors"):
                base, adapter, self.tuned = MERGED, None, True                       # merged model, no adapter folder
            else:
                self.tuned = os.environ.get("AGENTIX_READER_TUNED") == "1" or bool(adapter and os.path.exists(f"{adapter}/adapters.safetensors"))
            self.model, self.tok = load(base, adapter_path=adapter if self.tuned and adapter else None)
        self._mid = "|".join([self._model_files(), f"tuned={self.tuned}", f"max={max_tokens}", "" if self.url else base])
        # the fine-tuned model learned the format (short prompt); the base model zero-shot gets the full instructions
        self.read_sys, self.reply_sys = (FT_READ_SYSTEM, FT_REPLY_SYSTEM) if self.tuned else (READ_SYSTEM, REPLY_SYSTEM)

    @staticmethod
    def _model_files() -> str:
        """Identity of the model that answers: the sizes of its files (a different or re-trained model never reuses a cached read)."""
        parts = []
        for f in ("config.json", "model.safetensors", "tokenizer.json"):
            p = f"{MERGED}/{f}"
            parts.append(f"{f}:{os.path.getsize(p)}" if os.path.exists(p) else f"{f}:-")
        return ",".join(parts)

    def _gen(self, system: str, user: str) -> dict | None:
        """A read already made (any layout, any method, any run) is taken from the disk cache: no model call. Failures are never cached."""
        self.calls += 1
        key = hashlib.sha256(json.dumps([self._mid, system, user], ensure_ascii=False).encode()).hexdigest()
        path = os.path.join(CACHE_DIR, key[:2], key + ".json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    hit = json.load(f)
                self.cached += 1
                return hit
            except ValueError:
                pass
        out = self._gen_raw(system, user)
        self.errors += out is None
        if out is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                json.dump(out, f, ensure_ascii=False)
            os.replace(tmp, path)                       # atomic: three workers may write the same key
        return out

    def _gen_raw(self, system: str, user: str) -> dict | None:
        if self.url:
            return self._gen_http(system, user)
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        prompt = self.tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                              tokenize=False, add_generation_prompt=True)
        out = generate(self.model, self.tok, prompt=prompt, max_tokens=self.max_tokens, sampler=make_sampler(temp=0.0))
        try:
            return json.loads(out[out.find("{"): out.rfind("}") + 1])
        except ValueError:
            return None

    def _gen_http(self, system: str, user: str) -> dict | None:
        import urllib.request
        body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": self.max_tokens, "temperature": 0.0}
        req = urllib.request.Request(f"{self.url}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            out = json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"]
            return json.loads(out[out.find("{"): out.rfind("}") + 1])
        except (OSError, ValueError, KeyError):
            return None

    def read(self, report: dict) -> dict:
        user = json.dumps({"channel": report.get("channel"), "text": report["text"]}, ensure_ascii=False)
        return repair_read(self._gen(self.read_sys, user))

    def read_many(self, reports: list[dict]) -> dict[str, dict]:
        return {r["report_id"]: self.read(r) for r in reports}

    def read_replies(self, items: list[tuple[str, str | None]]) -> list[dict]:
        out = []
        for q, a in items:
            if not a:
                out.append(dict(EMPTY_REPLY))
                continue
            out.append(repair_reply(self._gen(self.reply_sys, json.dumps({"question": q, "answer": a}, ensure_ascii=False))))
        return out


_SHARED: LocalReader | None = None


def shared_local_reader() -> LocalReader:
    """One reader per process (loading a model per case would exhaust an 8 GB machine). Its `errors` count is read by the harness."""
    global _SHARED
    if _SHARED is None:
        _SHARED = LocalReader()
    return _SHARED
