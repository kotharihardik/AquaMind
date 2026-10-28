"""Document search (RAG) over the utility's documents: the SOP (rules) and the complaint/repair records.

BM25 keyword ranking: the corpus is small and full of names, places and ids, where keyword match works well and is
transparent. Every chunk has a citable id ("SOP-5", "REC-014") and a provenance: "policy" (trusted rules) or
"record" (useful facts; never treated as an instruction). Instruction-like sentences are removed before any text
reaches an agent.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_TOK = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "to", "in", "on", "and", "or", "is", "are", "for", "at", "by", "be", "it", "this",
         "with", "from", "as", "we", "our", "that", "must", "not", "no", "any", "all", "can", "may", "near"}
INSTRUCTION = re.compile(r"[^.]*(approval (already )?(given|granted)|ignore (all |previous |the )?(rules|instructions)|"
                         r"skip (the )?(evidence|checks)|system note|auto-?approv)[^.]*\.?", re.I)


def tokens(text: str) -> list[str]:
    return [t for t in _TOK.findall(text.lower()) if t not in _STOP]


def clean(text: str) -> str:
    return INSTRUCTION.sub("[instruction-like text removed]", text)


@dataclass
class Chunk:
    id: str
    text: str
    provenance: str


class Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.4, b: float = 0.75):
        self.chunks = chunks
        self.by_id = {c.id: c for c in chunks}
        self.docs = [Counter(tokens(c.text)) for c in chunks]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter(t for d in self.docs for t in d)
        self.idf = {t: math.log(1 + (len(chunks) - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 4, provenance: str | None = None) -> list[tuple[Chunk, float]]:
        q = tokens(query)
        scored = []
        for c, d, L in zip(self.chunks, self.docs, self.len):
            if provenance and c.provenance != provenance:
                continue
            s = sum(self.idf.get(t, 0) * d[t] * (self.k1 + 1) / (d[t] + self.k1 * (1 - self.b + self.b * L / self.avg))
                    for t in q if d.get(t))
            if s > 0:
                scored.append((c, s))
        return sorted(scored, key=lambda x: -x[1])[:k]


def load_corpus(sop_path: str, records_path: str) -> Retriever:
    chunks = []
    sop = open(sop_path).read()
    for m in re.finditer(r"^## (SOP-\d+)([^\n]*)\n(.*?)(?=^## |\Z)", sop, flags=re.S | re.M):
        chunks.append(Chunk(m.group(1), (m.group(2).strip() + ". " + m.group(3).strip()).replace("\n", " "), "policy"))
    for line in open(records_path):
        m = re.match(r"- (REC-\d+) \| ([^|]+) \| (.*)", line.strip())
        if m:
            chunks.append(Chunk(m.group(1), clean(f"{m.group(2).strip()}: {m.group(3).strip()}"), "record"))
    return Retriever(chunks)
