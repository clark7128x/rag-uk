from pathlib import Path

import pytest

from rag_uk.chunking import Chunk
from rag_uk.config import (
    BM25Config,
    ChunkingConfig,
    Config,
    EmbeddingProfile,
    EvalConfig,
    LLMConfig,
    RerankerConfig,
    RetrievalConfig,
    VectorConfig,
)


@pytest.fixture
def mk_chunk():
    def factory(text: str, ordinal: int = 0, doc_id: str = "d", section: str = "") -> Chunk:
        return Chunk(
            chunk_id=f"{doc_id}:{ordinal:04d}",
            doc_id=doc_id,
            ordinal=ordinal,
            section=section,
            text=text,
            char_start=0,
            char_end=len(text),
            paragraph_start=ordinal,
            paragraph_end=ordinal,
        )

    return factory


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    profiles = {
        "fake": EmbeddingProfile("fake", "fake/repo"),
        "fake2": EmbeddingProfile("fake2", "fake/repo2"),
    }
    return Config(
        corpus_dir=tmp_path / "corpus",
        corpus_glob=(".md",),
        cache_dir=tmp_path / "cache",
        chunking=ChunkingConfig(
            target_chars=200, overlap_chars=20, min_chars=0, hard_max_chars=400
        ),
        retrieval=RetrievalConfig(
            mode="hybrid",
            top_k=2,
            candidate_k=5,
            rrf_k=60,
            bm25=BM25Config(),
            vector=VectorConfig(model="fake", profiles=profiles),
            reranker=RerankerConfig(model="fake/reranker"),
        ),
        llm=LLMConfig(),
        eval=EvalConfig(dataset=tmp_path / "eval.jsonl", embedding_models=("fake", "fake2")),
        raw={},
    )
