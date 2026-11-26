from __future__ import annotations

import re
from typing import Iterable, Sequence

_ID = re.compile(r"^q-(\d+)$")


class LabelError(ValueError):
    pass


def next_id(existing_ids: Iterable[str]) -> str:
    numbers = [int(m.group(1)) for m in (_ID.match(i) for i in existing_ids) if m]
    return f"q-{max(numbers, default=0) + 1:04d}"


def make_record(
    question: str,
    gold_ids: Sequence[str],
    known_ids: set[str],
    existing_questions: set[str],
    record_id: str,
) -> dict:
    text = question.strip()
    if not text:
        raise LabelError("порожнє питання")
    if text.casefold() in existing_questions:
        raise LabelError("таке питання вже є в наборі")

    gold = list(dict.fromkeys(g.strip() for g in gold_ids if g.strip()))
    if not gold:
        raise LabelError("немає gold_chunk_ids")
    unknown = [g for g in gold if g not in known_ids]
    if unknown:
        raise LabelError(f"невідомі chunk_id: {unknown}")

    return {"id": record_id, "question": text, "gold_chunk_ids": gold}
