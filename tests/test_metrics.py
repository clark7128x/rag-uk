import pytest

from rag_uk.metrics import (
    citation_summary,
    latency_stats,
    percentile,
    recall_at_k,
    reciprocal_rank,
    retrieval_summary,
)


@pytest.mark.parametrize(
    "retrieved,gold,k,expected",
    [
        (["a", "b", "c"], ["a", "c"], 1, 0.5),
        (["a", "b", "c"], ["a", "c"], 3, 1.0),
        (["a", "b", "c"], ["a", "c"], 2, 0.5),
        (["a", "b"], ["a", "c"], 10, 0.5),
        (["a"], ["a", "a"], 1, 1.0),
        ([], ["a"], 5, 0.0),
        (["x", "y"], ["a"], 5, 0.0),
    ],
)
def test_recall_at_k(retrieved, gold, k, expected):
    assert recall_at_k(retrieved, gold, k) == pytest.approx(expected)


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_recall_invalid_k(k):
    with pytest.raises(ValueError):
        recall_at_k(["a"], ["a"], k)


def test_recall_empty_gold():
    with pytest.raises(ValueError):
        recall_at_k(["a"], [], 1)


@pytest.mark.parametrize(
    "retrieved,gold,expected",
    [(["a"], ["a"], 1.0), (["x", "a"], ["a"], 0.5), (["x"], ["a"], 0.0), ([], ["a"], 0.0)],
)
def test_reciprocal_rank(retrieved, gold, expected):
    assert reciprocal_rank(retrieved, gold) == pytest.approx(expected)


def test_reciprocal_rank_empty_gold():
    with pytest.raises(ValueError):
        reciprocal_rank(["a"], [])


def test_retrieval_summary_normal():
    rows = [
        {"retrieved": ["a", "b"], "gold_chunk_ids": ["a"]},
        {"retrieved": ["x", "y"], "gold_chunk_ids": ["a"]},
    ]
    summary = retrieval_summary(rows, [1, 3])
    assert summary["n"] == 2
    assert summary["recall@1"] == pytest.approx(0.5)
    assert summary["recall@3"] == pytest.approx(0.5)
    assert summary["mrr"] == pytest.approx(0.5)


def test_retrieval_summary_empty_returns_none():
    summary = retrieval_summary([], [1, 3, 5])
    assert summary["n"] == 0
    assert summary["recall@1"] is None and summary["mrr"] is None


def test_retrieval_summary_rejects_invalid_k():
    with pytest.raises(ValueError):
        retrieval_summary([{"retrieved": ["a"], "gold_chunk_ids": ["a"]}], [0])


def test_citation_summary():
    reports = [
        {"has_citations": True, "all_in_context": True, "unknown_citations": []},
        {"has_citations": True, "all_in_context": False, "unknown_citations": ["d:9999"]},
        {"has_citations": False, "all_in_context": False, "unknown_citations": []},
    ]
    summary = citation_summary(reports)
    assert summary["answers"] == 3
    assert summary["with_citations"] == 2
    assert summary["no_citation_answers"] == 1
    assert summary["answers_with_unknown_citation"] == 1
    assert summary["grounding_rate"] == pytest.approx(1 / 3)


def test_citation_summary_empty():
    assert citation_summary([])["grounding_rate"] is None


@pytest.mark.parametrize(
    "values,p,expected",
    [([0.1], 50, 0.1), ([0.1, 0.2, 0.3], 0, 0.1), ([0.1, 0.2, 0.3], 100, 0.3)],
)
def test_percentile_boundaries(values, p, expected):
    assert percentile(values, p) == pytest.approx(expected)


def test_percentile_empty_and_invalid():
    assert percentile([], 50) is None
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_latency_stats():
    assert latency_stats([]) == {"mean": None, "p50": None, "p95": None, "max": None}
    assert latency_stats([1.0, 3.0])["mean"] == pytest.approx(2.0)
