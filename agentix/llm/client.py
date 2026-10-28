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
def _discover_provider_keys(provider: str) -> list[str]:
    """Find all configured API keys for a provider dynamically from os.environ.
    Supports both exact names and suffixed indices (e.g. NVIDIA_API_KEY, NVIDIA_API_KEY_1..N, GEMINI_API_1..N).
    """
    prefix_map = {
        "nvidia": ["NVIDIA_API_KEY"],
        "gemini": ["GEMINI_API", "GEMINI_API_KEY"],
        "groq": ["GROQ_API_KEY"],
        "cerebras": ["CEREBRAS_API_KEY"],
        "openrouter": ["OPENROUTER_API_KEY"],
    }
    prefixes = prefix_map.get(provider, [f"{provider.upper()}_API_KEY"])
    keys = []
    
    # Check exact matches first in defined order
    for p in prefixes:
        val = os.environ.get(p)
        if val and val.strip():
            keys.append(val.strip())
            
    # Then discover all numbered or suffixed environment variables
    # e.g., NVIDIA_API_KEY_1, NVIDIA_API_KEY_2, ..., NVIDIA_API_KEY_50
    discovered = []
    for env_k, env_v in os.environ.items():
        if not env_v or not env_v.strip():
            continue
        for p in prefixes:
            if env_k.startswith(f"{p}_") or env_k.startswith(f"{p}"):
                if env_k not in prefixes and (env_k.startswith(f"{p}_") or env_k.startswith(f"{p}")):
                    discovered.append((env_k, env_v.strip()))
                    break

    # Sort numerically if ending in numbers, else alphabetically for deterministic ordering
    def _sort_key(item):
        k = item[0]
        parts = k.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return (parts[0], int(parts[1]))
        return (k, 0)

    discovered.sort(key=_sort_key)
    for _, v in discovered:
        keys.append(v)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            deduped.append(k)
    return deduped


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
        self.keys = _discover_provider_keys(self.provider)
        if not self.keys:
            raise RuntimeError(f"no API key for {self.provider} in the environment or .env")
        self._key_tags = [hashlib.sha256(k.encode()).hexdigest()[:8] for k in self.keys]
        self._k = (os.getpid() + random.randrange(len(self.keys))) % len(self.keys)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(".cache/rate", exist_ok=True)

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

    def _rate_file(self, k_idx: int) -> str:
        tag = self._key_tags[k_idx]
        return f".cache/rate/{self.provider}_{tag}.json"

    def _key_wait_estimate(self, k_idx: int) -> tuple[float, float]:
        base = 60.0 / RPM.get(self.provider, 20.0)
        rf = self._rate_file(k_idx)
        if not os.path.exists(rf):
            return 0.0, base
        try:
            with open(rf, "r") as f:
                st = json.loads(f.read() or "{}")
            gap = float(st.get("gap", base))
            wait = max(0.0, float(st.get("last", 0.0)) + gap - time.time())
            return wait, gap
        except Exception:
            return 0.0, base

    def _reserve_pace_key(self, k_idx: int) -> float:
        """Reserve a timeslot for the specific API key using virtual clock reservation.
        Releases the lock immediately so other processes are never blocked from reading/reserving slots.
        Returns the duration (seconds) this process needs to sleep before making the request.
        """
        base = 60.0 / RPM.get(self.provider, 20.0)
        rf = self._rate_file(k_idx)
        now = time.time()
        with open(rf, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            try:
                st = json.loads(f.read() or "{}")
            except ValueError:
                st = {}
            gap = max(base, float(st.get("gap", base)))
            last = float(st.get("last", 0.0))
            scheduled = max(now, last + gap)
            wait = scheduled - now
            st.update(last=scheduled, gap=gap)
            f.seek(0)
            f.truncate()
            f.write(json.dumps(st))
            fcntl.flock(f, fcntl.LOCK_UN)
        return max(0.0, wait)

    def _adapt_key(self, k_idx: int, throttled: bool) -> None:
        """Adapt the rate gap for the specific API key."""
        base = 60.0 / RPM.get(self.provider, 20.0)
        rf = self._rate_file(k_idx)
        with open(rf, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            try:
                st = json.loads(f.read() or "{}")
            except ValueError:
                st = {}
            gap = float(st.get("gap", base))
            if throttled:
                st["gap"] = min(2 * base, gap * 1.25)
                st["last"] = max(float(st.get("last", 0.0)), time.time() + 3.0)
            else:
                st["gap"] = max(base, gap * 0.9)
            f.seek(0)
            f.truncate()
            f.write(json.dumps(st))
            fcntl.flock(f, fcntl.LOCK_UN)

    def _ask(self, system, user, required):
        last, tries, throttled = None, 0, 0
        preferred_k = (os.getpid() + random.randrange(len(self.keys))) % len(self.keys)
        while tries < 4 and throttled < 60:
            # Pick the key with the smallest estimated wait time
            candidates = []
            for i in range(len(self.keys)):
                k_idx = (preferred_k + i) % len(self.keys)
                wait_s, _ = self._key_wait_estimate(k_idx)
                candidates.append((wait_s, k_idx))
            candidates.sort(key=lambda x: x[0])
            chosen_key_idx = candidates[0][1]

            wait = self._reserve_pace_key(chosen_key_idx)
            if wait > 0:
                time.sleep(wait)
                self.usage.waited += wait

            try:
                text, pt, ot = self._post(system, user, tries > 0, chosen_key_idx)
                self._adapt_key(chosen_key_idx, False)
                return _parse(text, required), pt, ot
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
                if e.code in (429, 503):
                    self._adapt_key(chosen_key_idx, True)
                    throttled += 1
                    self.usage.throttled += 1
                    # Shift preference immediately to next key
                    preferred_k = (chosen_key_idx + 1) % len(self.keys)
                    continue
                tries += 1
                time.sleep(2 ** tries)
            except (urllib.error.URLError, TimeoutError, ValueError, OSError, KeyError) as e:
                last = repr(e)[:200]
                tries += 1
                time.sleep(2 ** tries)
        return {"_error": last}, 0, 0

    def _post(self, system, user, retry: bool, key_idx: int):
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
                                              "Authorization": f"Bearer {self.keys[key_idx]}"})
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
