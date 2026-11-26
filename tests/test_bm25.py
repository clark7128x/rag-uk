import pytest

from rag_uk.bm25 import BM25Index, tokenize
from rag_uk.config import BM25Config, ConfigError


def build(mk_chunk, texts, cfg=None):
    chunks = [mk_chunk(text, ordinal=i) for i, text in enumerate(texts)]
    return BM25Index.build(chunks, cfg or BM25Config(), prefix_section=False)


def test_ranks_matching_documents_only(mk_chunk):
    index = build(
        mk_chunk,
        [
            "Оплата здійснюється протягом десяти днів",
            "Розірвання договору за згодою сторін",
            "Оплата штрафу та пені",
        ],
    )
    hits = index.search("оплата", 5)
    assert {cid for cid, _ in hits} == {"d:0000", "d:0002"}
    assert all(score > 0 for _, score in hits)


def test_shorter_document_scores_higher(mk_chunk):
    index = build(mk_chunk, ["Оплата штраф та пеня і ще багато слів тут", "Оплата штрафу"])
    assert index.search("оплата", 2)[0][0] == "d:0001"


def test_apostrophe_normalization():
    assert tokenize("п’ять П'ЯТЬ ʼять") == ["п'ять", "п'ять", "'ять"]


def test_section_prefix_is_indexed(mk_chunk):
    chunks = [mk_chunk("текст без збігу", ordinal=0, section="Оплата")]
    index = BM25Index.build(chunks, BM25Config(), prefix_section=True)
    assert index.search("оплата", 1)[0][0] == "d:0000"


@pytest.mark.parametrize("query", ["", "   ", "невідоме слово"])
def test_queries_without_matches(mk_chunk, query):
    assert build(mk_chunk, ["Оплата штрафу"]).search(query, 5) == []


def test_empty_index(mk_chunk):
    assert build(mk_chunk, []).search("оплата", 5) == []


def test_k_boundaries(mk_chunk):
    index = build(mk_chunk, ["Оплата", "Оплата штрафу"])
    assert index.search("оплата", 0) == []
    assert len(index.search("оплата", 100)) == 2
    assert len(index.search("оплата", 1)) == 1


def test_ties_are_deterministic(mk_chunk):
    index = build(mk_chunk, ["Оплата штрафу", "Оплата штрафу"])
    first = index.search("оплата", 2)
    assert [cid for cid, _ in first] == ["d:0000", "d:0001"]
    assert first == index.search("оплата", 2)


def test_documents_without_tokens(mk_chunk):
    assert build(mk_chunk, ["", ""]).search("оплата", 5) == []


@pytest.mark.parametrize("params", [{"k1": 0}, {"k1": -1}, {"b": -0.1}, {"b": 1.5}])
def test_invalid_bm25_params(params):
    with pytest.raises(ConfigError):
        BM25Config(**params)
