"""Prompted LLM reader (reader-2.0 / reply-2.0): a large model with instructions, no fine-tuning.

It is the "large prompted model" reference of the fine-tuning study and the stand-in reader when the fine-tuned
small model is not loaded. It reads the messages of one case in ONE call (each message stays separate, keyed by its
id) to keep the number of calls per case small. Raw text goes in; only schema-checked fields come out (quarantine).
"""
from __future__ import annotations

import json

from .prompts import READ_SYSTEM, REPLY_SYSTEM
from .schema import EMPTY_REPLY, repair_read, repair_reply


class LLMReader:
    name = "llm_prompted"

    def __init__(self, llm, batch: int = 4, parallel: int = 2):
        self.llm, self.batch, self.parallel = llm, batch, parallel

    def _chunk(self, chunk: list[dict]) -> dict[str, dict]:
        user = ("Read each message separately. Return {\"reads\": {\"<id>\": <reader-2.0 object>, ...}} with one "
                "entry per id.\n\n" + json.dumps([{"id": r["report_id"], "channel": r.get("channel"),
                                                    "text": r["text"]} for r in chunk], ensure_ascii=False))
        d = self.llm.json(READ_SYSTEM, user, ["reads"], tag="reader")
        reads = d.get("reads") if isinstance(d.get("reads"), dict) else {}
        return {r["report_id"]: repair_read(reads.get(r["report_id"])) for r in chunk}

    def read_many(self, reports: list[dict]) -> dict[str, dict]:
        """Small batches in parallel: the time is in the output tokens, so 4 short answers beat 1 long one."""
        from concurrent.futures import ThreadPoolExecutor
        chunks = [reports[i:i + self.batch] for i in range(0, len(reports), self.batch)]
        out = {}
        with ThreadPoolExecutor(self.parallel) as ex:
            for part in ex.map(self._chunk, chunks):
                out.update(part)
        return out

    def read(self, report: dict) -> dict:
        return self.read_many([report])[report["report_id"]]

    def read_replies(self, items: list[tuple[str, str | None]]) -> list[dict]:
        """items: [(question, answer text or None)] -> reply-2.0 objects."""
        todo = [(k, q, a) for k, (q, a) in enumerate(items) if a]
        out = [dict(EMPTY_REPLY) for _ in items]
        if not todo:
            return out
        user = ("Read each call separately. Return {\"replies\": {\"<n>\": <reply-2.0 object>, ...}}.\n\n" +
                json.dumps([{"n": str(k), "question": q, "answer": a} for k, q, a in todo], ensure_ascii=False))
        d = self.llm.json(REPLY_SYSTEM, user, ["replies"], tag="reply_reader")
        got = d.get("replies") if isinstance(d.get("replies"), dict) else {}
        for k, _, _ in todo:
            out[k] = repair_reply(got.get(str(k)))
        return out
