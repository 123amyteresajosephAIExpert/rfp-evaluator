"""Ask-the-proposals tool: Retrieval-Augmented Generation over supplier documents.

1. Chunk  - each proposal is split into ~120-word overlapping chunks that remember
            their supplier and page number.
2. Retrieve - BM25 keyword ranking (pure Python, no extra download), taking the best
            chunks *per supplier* so comparisons cover everyone in scope.
3. Generate - the LLM answers using only those numbered excerpts and cites them [n].
            Without an AI key, the best excerpts are shown instead (extractive answer).

Why BM25 rather than embeddings here: proposals are short, analyst questions are
specific ("ISO 27001", "SuccessFactors", "service credits"), keyword retrieval is
exact for such terms, and it needs no embedding model on the free hosting tier.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from . import config, db
from .llm_client import LLMClient, LLMError

_STOP = set("a an and are as at be by for from has have how in is it its of on or that the this to was were what "
            "which who will with does do did their they them about any all can our we you your there".split())


@dataclass
class Chunk:
    chunk_id: int
    supplier: str
    page: int
    text: str


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens with a light prefix stem, so 'certified' matches 'certification'."""
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t[:7] if t.isalpha() and len(t) > 7 else t for t in toks if t not in _STOP]


def chunk_document(supplier: str, text: str, start_id: int = 0) -> list[Chunk]:
    chunks: list[Chunk] = []
    pages = re.split(r"\[Page (\d+)\]", text)
    # re.split with a group gives ["", "1", page1, "2", page2 ...]
    page_texts = [(int(pages[i]), pages[i + 1]) for i in range(1, len(pages) - 1, 2)] or [(1, text)]
    step = config.CHUNK_WORDS - config.CHUNK_OVERLAP_WORDS
    for page, body in page_texts:
        words = body.split()
        for i in range(0, max(1, len(words)), step):
            piece = " ".join(words[i:i + config.CHUNK_WORDS])
            if len(piece) > 40:
                chunks.append(Chunk(start_id + len(chunks), supplier, page, piece))
            if i + config.CHUNK_WORDS >= len(words):
                break
    return chunks


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avgdl = sum(len(d) for d in docs) / max(1, len(docs))
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def score(self, query: list[str], i: int) -> float:
        tf, dl, s = self.tf[i], len(self.docs[i]), 0.0
        for q in query:
            if q in tf:
                f = tf[q]
                s += self.idf.get(q, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s


def build_index(documents: dict[str, str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for supplier, text in documents.items():
        chunks += chunk_document(supplier, text, start_id=len(chunks))
    return chunks


def retrieve(question: str, chunks: list[Chunk], suppliers: list[str] | None = None,
             per_supplier: int = config.RAG_TOP_K_PER_SUPPLIER) -> list[tuple[Chunk, float]]:
    pool = [c for c in chunks if not suppliers or c.supplier in suppliers]
    if not pool:
        return []
    bm25 = BM25([tokenize(c.text) for c in pool])
    q = tokenize(question)
    scored = sorted(((c, bm25.score(q, i)) for i, c in enumerate(pool)), key=lambda x: (-x[1], x[0].chunk_id))
    picked, counts = [], Counter()
    for c, sc in scored:
        if sc <= 0:
            continue
        if counts[c.supplier] < per_supplier:
            picked.append((c, sc))
            counts[c.supplier] += 1
    return picked


def answer(question: str, run_id: str, suppliers: list[str] | None, client: LLMClient | None, db_path=None) -> dict:
    docs = db.get_run_documents(run_id, db_path)
    if not docs:
        return {"answer": "No proposal text is stored for this run.", "sources": [], "mode": "none"}
    chunks = build_index({k: v["text"] for k, v in docs.items()})
    hits = retrieve(question, chunks, suppliers)
    if not hits:
        return {"answer": "None of the selected proposals mention anything matching this question. "
                          "Try different words (for example 'ISO', 'price', 'timeline').",
                "sources": [], "mode": "none"}
    sources = [{"n": i + 1, "supplier": c.supplier, "page": c.page, "text": c.text, "score": round(sc, 3)}
               for i, (c, sc) in enumerate(hits)]
    if client is None:
        return {"answer": "Offline mode: showing the most relevant passages from each proposal "
                          "(add an AI key for a written answer).", "sources": sources, "mode": "extractive"}
    excerpts = "\n\n".join(f"[{s['n']}] {s['supplier']} (page {s['page']}): {s['text']}" for s in sources)
    system = db.get_prompt("qa_system", db_path)["text"]
    user = db.get_prompt("qa_user", db_path)["text"].format(question=question, excerpts=excerpts)
    try:
        text = client.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=900)
    except LLMError as exc:
        return {"answer": f"{exc} Showing the retrieved passages instead.", "sources": sources, "mode": "extractive"}
    return {"answer": text, "sources": sources, "mode": "llm"}
