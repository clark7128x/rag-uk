from __future__ import annotations

from typing import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    if k < 1:
        raise ValueError("rrf_k має бути >= 1")
    if top_k is not None and top_k < 0:
        raise ValueError("top_k має бути >= 0")

    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, chunk_id in enumerate(ranking, start=1):
            if chunk_id in seen:
                continue  # дубль усередині одного списку не додає балів
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered if top_k is None else ordered[:top_k]
