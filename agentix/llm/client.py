"""One small client for the free LLM endpoints we use (NVIDIA NIM, Google Gemini), with a disk cache.

- Every LLM method in a comparison uses the SAME model (fair comparison); the model is chosen by the caller.
- Answers are cached on disk (same prompt -> same answer): reruns are free and repeatable.
- Calls, tokens and seconds are counted, so cost and time per case can be reported.
- Free tiers throttle (HTTP 429): we wait and retry with jitter. Keys come from `.env` and are never printed.

Model names: "<provider>:<model>" with provider nvidia | gemini | groq | cerebras | openrouter, e.g. "nvidia:nvidia/nemotron-3-ultra-550b-a55b".
Add GROQ_API_KEY / CEREBRAS_API_KEY / OPENROUTER_API_KEY to .env to use the faster providers (free tiers); nothing else changes.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

ENDPOINTS = {"nvidia": "https://integrate.api.nvidia.com/v1/chat/completions",
             "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
             "groq": "https://api.groq.com/openai/v1/chat/completions",              # free tier, very fast: e.g. "groq:openai/gpt-oss-120b", "groq:llama-3.3-70b-versatile"
             "cerebras": "https://api.cerebras.ai/v1/chat/completions",              # free tier, very fast: e.g. "cerebras:gpt-oss-120b", "cerebras:llama-3.3-70b"
             "openrouter": "https://openrouter.ai/api/v1/chat/completions"}          # free models end in ":free"
RPM = {"nvidia": float(os.environ.get("NVIDIA_RPM", "20")), "gemini": float(os.environ.get("GEMINI_RPM", "10")),
       "groq": float(os.environ.get("GROQ_RPM", "25")), "cerebras": float(os.environ.get("CEREBRAS_RPM", "25")),
       "openrouter": float(os.environ.get("OPENROUTER_RPM", "15"))}   # free tiers
KEYS = {"nvidia": ["NVIDIA_API_KEY"], "gemini": ["GEMINI_API_1", "GEMINI_API_2", "GEMINI_API_KEY"], "groq": ["GROQ_API_KEY"],
        "cerebras": ["CEREBRAS_API_KEY"], "openrouter": ["OPENROUTER_API_KEY"]}
AGENT_MODEL = "nvidia:nvidia/nemotron-3-ultra-550b-a55b"      # every agent / LLM baseline (same model)
WRITER_MODEL = "gemini:gemini-3.1-flash-lite"                  # environment text (different family)


def load_dotenv(path: str = ".env") -> None:
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    errors: int = 0
    throttled: int = 0
    waited: float = 0.0            # seconds spent waiting for the free tier's rate limit (pacing + 429 back-off)


@dataclass
class LLM:
    model: str = AGENT_MODEL
    temperature: float = 0.0
    cache_dir: str = ".cache/llm"
    timeout_s: float = 120.0
    max_tokens: int = 2048
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self):
        load_dotenv()
        self.provider, self.name = self.model.split(":", 1)
        self.keys = [os.environ[k] for k in KEYS[self.provider] if os.environ.get(k)]
        if not self.keys:
            raise RuntimeError(f"no API key for {self.provider} in the environment or .env")
        self._k = random.randrange(len(self.keys))
        os.makedirs(self.cache_dir, exist_ok=True)

    def json(self, system: str, user: str, required: list[str] | None = None, tag: str = "") -> dict:
        """Ask for ONE JSON object. Returns the parsed dict, or {"_error": ...}."""
        key = hashlib.sha256(json.dumps([self.model, self.temperature, system, user], sort_keys=True).encode()).hexdigest()
        path = os.path.join(self.cache_dir, key[:2], key + ".json")
        if os.path.exists(path):
            self.usage.cached += 1
            return json.load(open(path))["answer"]
        t0 = time.time()
        answer, pt, ot = self._ask(system, user, required or [])
        self.usage.calls += 1
        self.usage.seconds += time.time() - t0
        self.usage.prompt_tokens += pt
        self.usage.output_tokens += ot
        if "_error" in answer:
            self.usage.errors += 1
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump({"tag": tag, "model": self.model, "answer": answer}, open(path, "w"))
        return answer

    def _pace(self) -> None:
        """One shared clock for all worker processes. The free tiers sustain only a few calls a minute and answer 429
        to bursts, but a 429 is cheap and faster attempts still give more successes per minute (measured), so the gap
        only widens a little after a 429 (x1.25, at most 2x the base gap 60/RPM) and recovers after a success. The lock is held while sleeping, so waiting workers queue in order."""
        base = 60.0 / RPM[self.provider]
        os.makedirs(".cache/rate", exist_ok=True)
        with open(f".cache/rate/{self.provider}.json", "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            try:
                st = json.loads(f.read() or "{}")
            except ValueError:
                st = {}
            gap = max(base, float(st.get("gap", base)))
            wait = max(0.0, float(st.get("last", 0.0)) + gap - time.time())
            if wait:
                time.sleep(wait)
                self.usage.waited += wait
            st.update(last=time.time(), gap=gap)
            f.seek(0)
            f.truncate()
            f.write(json.dumps(st))

    def _adapt(self, throttled: bool) -> None:
        base = 60.0 / RPM[self.provider]
        with open(f".cache/rate/{self.provider}.json", "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            try:
                st = json.loads(f.read() or "{}")
            except ValueError:
                st = {}
            gap = float(st.get("gap", base))
            st["gap"] = min(2 * base, gap * 1.25) if throttled else max(base, gap * 0.9)
            f.seek(0)
            f.truncate()
            f.write(json.dumps(st))

    def _ask(self, system, user, required):
        last, tries, throttled = None, 0, 0
        while tries < 4 and throttled < 60:
            try:
                self._pace()
                text, pt, ot = self._post(system, user, tries > 0)
                self._adapt(False)
                return _parse(text, required), pt, ot
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
                if e.code in (429, 503):
                    self._adapt(True)
                    throttled += 1
                    self.usage.throttled += 1
                    self._k = (self._k + 1) % len(self.keys)
                    continue                        # the shared pacer spaces the retry (no extra private back-off)
                tries += 1
                time.sleep(2 ** tries)
            except (urllib.error.URLError, TimeoutError, ValueError, OSError, KeyError) as e:
                last = repr(e)[:200]
                tries += 1
                time.sleep(2 ** tries)
        return {"_error": last}, 0, 0

    def _post(self, system, user, retry: bool):
        sys_msg = system + "\n\nAnswer with ONE JSON object only (no markdown)."
        if retry:
            sys_msg = "Keep any reasoning very short.\n" + sys_msg
        body = {"model": self.name, "temperature": self.temperature, "max_tokens": self.max_tokens * (2 if retry else 1),
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]}
        if "nemotron-3" in self.name:          # hidden reasoning off: 5 s instead of 40 s per decision, same choices
            body["chat_template_kwargs"] = {"enable_thinking": False}
        req = urllib.request.Request(ENDPOINTS[self.provider], data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.keys[self._k]}"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            d = json.loads(r.read())
        u = d.get("usage") or {}
        return d["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def _parse(text: str, required: list[str]) -> dict:
    if not text:
        raise ValueError("empty answer")
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
    missing = [k for k in required if k not in obj]
    if missing:
        raise ValueError(f"missing keys {missing}")
    return obj
