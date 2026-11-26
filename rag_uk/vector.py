from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from .calllog import CallLog
from .chunking import Chunk
from .config import EmbeddingProfile


class EmbeddingModel:
    def __init__(
        self,
        profile: EmbeddingProfile,
        device: str = "cpu",
        batch_size: int = 32,
        call_log: CallLog | None = None,
    ):
        self.profile = profile
        self.device = device
        self.batch_size = batch_size
        self.call_log = call_log
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Векторний режим потребує sentence-transformers: pip install -e '.[vector]'"
                ) from exc
            self._model = SentenceTransformer(self.profile.repo, device=self.device)
        return self._model

    def _count_tokens(self, model, texts: list[str]) -> int | None:
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            return None
        try:
            encoded = tokenizer(
                texts,
                truncation=True,
                max_length=getattr(model, "max_seq_length", None),
            )
            return sum(len(ids) for ids in encoded["input_ids"])
        except Exception:
            return None

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        model = self._load()
        prefix = self.profile.query_prefix if kind == "query" else self.profile.passage_prefix
        payload = [prefix + t for t in texts]
        started = perf_counter()
        matrix = model.encode(
            payload,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        latency = perf_counter() - started
        if self.call_log:
            self.call_log.record(
                kind="embedding",
                model=self.profile.repo,
                input_tokens=self._count_tokens(model, payload),
                output_tokens=0,
                latency_s=latency,
                extra={"texts": len(texts), "purpose": kind},
            )
        return matrix


def _fingerprint(chunks: list[Chunk], profile: EmbeddingProfile, prefix_section: bool) -> str:
    digest = hashlib.sha256()
    digest.update(profile.repo.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(profile.passage_prefix.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(str(prefix_section).encode("utf-8"))
    for chunk in chunks:
        digest.update(b"\x00")
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(chunk.index_text(prefix_section).encode("utf-8"))
    return digest.hexdigest()[:32]


class VectorIndex:
    def __init__(self, ids: list[str], matrix: np.ndarray):
        self.ids = ids
        self.matrix = matrix

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        encoder,
        cache_dir: Path | None,
        prefix_section: bool,
    ) -> "VectorIndex":
        ids = [c.chunk_id for c in chunks]
        if not chunks:
            return cls(ids, np.zeros((0, 0), dtype=np.float32))

        path: Path | None = None
        if cache_dir is not None:
            name = _fingerprint(chunks, encoder.profile, prefix_section)
            path = Path(cache_dir) / encoder.profile.name / f"{name}.npy"
            if path.exists():
                matrix = np.load(path)
                if matrix.shape[0] == len(chunks):
                    return cls(ids, matrix)

        matrix = encoder.encode([c.index_text(prefix_section) for c in chunks], kind="passage")
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, matrix)
        return cls(ids, matrix)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if k <= 0 or not self.ids:
            return []
        query = np.asarray(query_vector, dtype=np.float32)
        if query.shape[0] != self.matrix.shape[1]:
            raise ValueError(
                f"Розмірність запиту {query.shape[0]} не збігається з індексом {self.matrix.shape[1]}"
            )
        norm = float(np.linalg.norm(query))
        if norm > 0:
            query = query / norm
        scores = self.matrix @ query
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.ids[int(i)], float(scores[int(i)])) for i in order]
