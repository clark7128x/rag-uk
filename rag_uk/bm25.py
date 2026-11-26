from __future__ import annotations

import math
import re
from collections import defaultdict

from .chunking import Chunk
from .config import BM25Config

_TOKEN = re.compile(r"\w+(?:['’ʼ]\w+)*", re.UNICODE)
_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'"})


def tokenize(text: str) -> list[str]:
    return [t.translate(_APOSTROPHES) for t in _TOKEN.findall(text.lower())]


class BM25Index:
    def __init__(self, ids: list[str], token_docs: list[list[str]], cfg: BM25Config):
        self.ids = ids
        self.cfg = cfg
        self.n_docs = len(ids)
        self.lengths = [len(tokens) for tokens in token_docs]
        self.avgdl = (sum(self.lengths) / self.n_docs) if self.n_docs else 0.0
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        for index, tokens in enumerate(token_docs):
            for token in tokens:
                self.postings[token][index] = self.postings[token].get(index, 0) + 1

    @classmethod
    def build(cls, chunks: list[Chunk], cfg: BM25Config, prefix_section: bool) -> "BM25Index":
        return cls(
            [c.chunk_id for c in chunks],
            [tokenize(c.index_text(prefix_section)) for c in chunks],
            cfg,
        )

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if k <= 0 or self.n_docs == 0:
            return []
        terms = tokenize(query)
        if not terms:
            return []

        k1, b = self.cfg.k1, self.cfg.b
        scores: dict[int, float] = {}
        for term in dict.fromkeys(terms):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = math.log(1 + (self.n_docs - len(posting) + 0.5) / (len(posting) + 0.5))
            for index, tf in posting.items():
                norm = (
                    k1 * (1 - b + b * self.lengths[index] / self.avgdl)
                    if self.avgdl > 0
                    else k1
                )
                scores[index] = scores.get(index, 0.0) + idf * tf * (k1 + 1) / (tf + norm)

        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], self.ids[kv[0]]))
        return [(self.ids[index], score) for index, score in ordered[:k]]
