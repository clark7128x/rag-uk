from __future__ import annotations

import math
from typing import Sequence


def _check_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k має бути цілим числом > 0")


def _check_gold(gold: Sequence[str]) -> set[str]:
    unique = set(gold)
    if not unique:
        raise ValueError("gold_chunk_ids не може бути порожнім")
    return unique


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    _check_k(k)
    relevant = _check_gold(gold)
    hits = relevant.intersection(retrieved[:k])
    return len(hits) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    relevant = _check_gold(gold)
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def retrieval_summary(rows: list[dict], k_values: Sequence[int]) -> dict:
    for k in k_values:
        _check_k(k)
    if not rows:
        return {"n": 0, **{f"recall@{k}": None for k in k_values}, "mrr": None}
    summary = {"n": len(rows)}
    for k in k_values:
        values = [recall_at_k(r["retrieved"], r["gold_chunk_ids"], k) for r in rows]
        summary[f"recall@{k}"] = sum(values) / len(values)
    rr = [reciprocal_rank(r["retrieved"], r["gold_chunk_ids"]) for r in rows]
    summary["mrr"] = sum(rr) / len(rr)
    return summary


def citation_summary(reports: list[dict]) -> dict:
    n = len(reports)
    with_citations = sum(1 for r in reports if r["has_citations"])
    grounded = sum(1 for r in reports if r["all_in_context"])
    return {
        "answers": n,
        "with_citations": with_citations,
        "no_citation_answers": n - with_citations,
        "answers_with_unknown_citation": sum(1 for r in reports if r["unknown_citations"]),
        "all_citations_in_context": grounded,
        "grounding_rate": grounded / n if n else None,
    }


def percentile(values: Sequence[float], p: float) -> float | None:
    if not 0 <= p <= 100:
        raise ValueError("p має бути в [0, 100]")
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(p / 100 * len(ordered)) - 1)
    return float(ordered[index])


def latency_stats(values: Sequence[float]) -> dict:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values),
    }
