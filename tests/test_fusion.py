import pytest

from rag_uk.fusion import reciprocal_rank_fusion


def test_merges_two_rankings():
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "a"]], k=60)
    assert [cid for cid, _ in fused] == ["a", "c", "b"]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_duplicate_inside_one_ranking_counts_once():
    fused = dict(reciprocal_rank_fusion([["a", "a", "b"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 63)


def test_tie_breaks_by_id():
    fused = reciprocal_rank_fusion([["b", "a"], ["a", "b"]], k=60)
    assert [cid for cid, _ in fused] == ["a", "b"]


def test_empty_inputs():
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_top_k_truncation():
    assert len(reciprocal_rank_fusion([["a", "b", "c"]], k=60, top_k=2)) == 2
    assert reciprocal_rank_fusion([["a"]], k=60, top_k=0) == []


def test_k_equal_one():
    assert reciprocal_rank_fusion([["a"]], k=1)[0][1] == pytest.approx(0.5)


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"k": -5}, {"top_k": -1}])
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], **kwargs)
