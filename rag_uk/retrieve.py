from __future__ import annotations

from dataclasses import dataclass

from .bm25 import BM25Index
from .calllog import CallLog
from .chunking import Chunk
from .config import RETRIEVAL_MODES, Config
from .fusion import reciprocal_rank_fusion
from .rerank import CrossEncoderReranker
from .vector import EmbeddingModel, VectorIndex


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    source: str


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        cfg: Config,
        *,
        encoder=None,
        reranker=None,
        call_log: CallLog | None = None,
        embedding_model: str | None = None,
    ):
        self.cfg = cfg
        self.chunks = list(chunks)
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.profile_name = embedding_model or cfg.retrieval.vector.model
        self.call_log = call_log
        self._encoder = encoder
        self._reranker = reranker
        self._vector: VectorIndex | None = None
        self._bm25 = BM25Index.build(
            self.chunks, cfg.retrieval.bm25, cfg.chunking.prefix_section
        )

    def _vector_index(self) -> VectorIndex:
        if self._vector is None:
            if self._encoder is None:
                self._encoder = EmbeddingModel(
                    self.cfg.retrieval.vector.profile(self.profile_name),
                    device=self.cfg.retrieval.vector.device,
                    batch_size=self.cfg.retrieval.vector.batch_size,
                    call_log=self.call_log,
                )
            self._vector = VectorIndex.build(
                self.chunks,
                self._encoder,
                self.cfg.cache_dir / "embeddings",
                self.cfg.chunking.prefix_section,
            )
        return self._vector

    def _bm25_ids(self, query: str, k: int) -> list[tuple[str, float]]:
        return self._bm25.search(query, k)

    def _vector_ids(self, query: str, k: int) -> list[tuple[str, float]]:
        index = self._vector_index()
        if not self.chunks:
            return []
        query_vector = self._encoder.encode([query], kind="query")[0]
        return index.search(query_vector, k)

    def search(
        self, query: str, mode: str | None = None, top_k: int | None = None
    ) -> list[Hit]:
        mode = mode or self.cfg.retrieval.mode
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"Невідомий режим пошуку: {mode}. Доступні: {RETRIEVAL_MODES}")
        top_k = self.cfg.retrieval.top_k if top_k is None else top_k
        if top_k <= 0:
            raise ValueError("top_k має бути > 0")
        if not query or not query.strip() or not self.chunks:
            return []

        if mode == "bm25":
            return [Hit(self.by_id[cid], s, "bm25") for cid, s in self._bm25_ids(query, top_k)]
        if mode == "vector":
            return [Hit(self.by_id[cid], s, "vector") for cid, s in self._vector_ids(query, top_k)]

        candidates = self.cfg.retrieval.candidate_k
        fused = reciprocal_rank_fusion(
            [
                [cid for cid, _ in self._bm25_ids(query, candidates)],
                [cid for cid, _ in self._vector_ids(query, candidates)],
            ],
            k=self.cfg.retrieval.rrf_k,
            top_k=candidates,
        )
        if mode == "hybrid":
            return [Hit(self.by_id[cid], s, "rrf") for cid, s in fused[:top_k]]

        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self.cfg.retrieval.reranker, self.call_log)
        scored = self._reranker.rerank(
            query,
            [self.by_id[cid] for cid, _ in fused],
            top_n=top_k,
            prefix_section=self.cfg.chunking.prefix_section,
        )
        return [Hit(self.by_id[cid], s, "rerank") for cid, s in scored]
