import numpy as np
import pytest

from rag_uk.config import EmbeddingProfile
from rag_uk.retrieve import Retriever


class FakeEncoder:
    def __init__(self, name="fake"):
        self.profile = EmbeddingProfile(name, f"{name}/repo")
        self.calls = 0

    def encode(self, texts, kind):
        self.calls += 1
        if not texts:
            return np.zeros((0, 2), dtype=np.float32)
        rows = [
            [float(t.lower().count("оплата")), float(t.lower().count("розірвання"))]
            for t in texts
        ]
        return np.asarray(rows, dtype=np.float32)


class ExplodingEncoder:
    profile = EmbeddingProfile("boom", "boom/repo")

    def encode(self, texts, kind):
        raise AssertionError("BM25 не має викликати енкодер")


class FakeReranker:
    def rerank(self, query, chunks, top_n, prefix_section=False):
        scored = [(c.chunk_id, float(len(c.text))) for c in chunks]
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored[:top_n]


@pytest.fixture
def chunks(mk_chunk):
    return [
        mk_chunk("Оплата штрафу", ordinal=0),
        mk_chunk("Розірвання договору за згодою сторін", ordinal=1),
        mk_chunk("Оплата здійснюється протягом десяти банківських днів", ordinal=2),
    ]


def test_bm25_mode_does_not_touch_encoder(chunks, cfg):
    retriever = Retriever(chunks, cfg, encoder=ExplodingEncoder())
    hits = retriever.search("оплата", mode="bm25")
    assert hits and all(h.source == "bm25" for h in hits)
    assert {h.chunk.chunk_id for h in hits} == {"d:0000", "d:0002"}


def test_vector_mode(chunks, cfg):
    retriever = Retriever(chunks, cfg, encoder=FakeEncoder())
    hits = retriever.search("оплата", mode="vector")
    assert len(hits) == cfg.retrieval.top_k
    assert all(h.source == "vector" for h in hits)
    # Обидва чанки з «оплата» мають однаковий косинус, порядок вирішує tie-break за позицією.
    assert [h.chunk.chunk_id for h in hits] == ["d:0000", "d:0002"]


def test_hybrid_mode_uses_rrf(chunks, cfg):
    retriever = Retriever(chunks, cfg, encoder=FakeEncoder())
    hits = retriever.search("оплата", mode="hybrid")
    assert hits and all(h.source == "rrf" for h in hits)


def test_hybrid_rerank_mode_uses_reranker_order(chunks, cfg):
    retriever = Retriever(chunks, cfg, encoder=FakeEncoder(), reranker=FakeReranker())
    hits = retriever.search("оплата", mode="hybrid_rerank")
    assert all(h.source == "rerank" for h in hits)
    assert hits[0].chunk.chunk_id == "d:0002"
    assert len(hits) == cfg.retrieval.top_k


def test_top_k_override(chunks, cfg):
    retriever = Retriever(chunks, cfg, encoder=FakeEncoder())
    assert len(retriever.search("оплата", mode="bm25", top_k=1)) == 1


def test_embedding_profile_selection(chunks, cfg):
    encoder = FakeEncoder("fake2")
    retriever = Retriever(chunks, cfg, encoder=encoder, embedding_model="fake2")
    retriever.search("оплата", mode="vector")
    assert retriever.profile_name == "fake2"


def test_unknown_mode(chunks, cfg):
    with pytest.raises(ValueError):
        Retriever(chunks, cfg).search("оплата", mode="magic")


def test_invalid_top_k(chunks, cfg):
    with pytest.raises(ValueError):
        Retriever(chunks, cfg).search("оплата", mode="bm25", top_k=0)


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query(chunks, cfg, query):
    assert Retriever(chunks, cfg).search(query, mode="bm25") == []


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid", "hybrid_rerank"])
def test_empty_corpus(cfg, mode):
    retriever = Retriever([], cfg, encoder=ExplodingEncoder())
    assert retriever.search("оплата", mode=mode) == []
