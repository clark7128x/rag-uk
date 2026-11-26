#!/usr/bin/env python3
"""Пропонує кандидати питань через LLM. Не пише в data/eval.jsonl."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_uk.calllog import CallLog
from rag_uk.config import ConfigError, load_config
from rag_uk.corpus import ensure_chunks
from rag_uk.jsonio import append_jsonl, read_jsonl
from rag_uk.llm import LLMClient, LLMError

PROMPT = (
    "Прочитай абзац документа українською мовою.\n"
    "Сформулюй одне питання, відповідь на яке міститься повністю в цьому абзаці.\n"
    "Питання має бути зрозумілим без самого абзацу: не використовуй займенники замість назв.\n"
    "Поверни лише текст питання одним рядком, без лапок і нумерації.\n\n"
    "Абзац:\n{text}"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--doc")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--out", default="data/eval_candidates.jsonl")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        _, paragraphs, _ = ensure_chunks(cfg)
    except ConfigError as exc:
        print(f"Конфіг: {exc}", file=sys.stderr)
        return 2
    if not cfg.llm.configured:
        print("Не задано LLM_API_BASE або LLM_MODEL", file=sys.stderr)
        return 2

    if args.doc:
        paragraphs = [p for p in paragraphs if p.doc_id == args.doc]
    selected = [p for p in paragraphs if p.chunk_ids][: args.limit]

    existing = read_jsonl(args.out) if Path(args.out).exists() else []
    counter = len(existing)
    call_log = CallLog(cfg.eval.call_log)
    llm = LLMClient(cfg.llm, call_log)

    written = 0
    for paragraph in selected:
        messages = [{"role": "user", "content": PROMPT.format(text=paragraph.text)}]
        try:
            result = llm.chat(messages)
        except LLMError as exc:
            print(f"{paragraph.paragraph_id}: {exc}", file=sys.stderr)
            continue
        question = next((line.strip() for line in result.text.splitlines() if line.strip()), "")
        question = question.strip("\"«»' ")
        if not question:
            continue
        counter += 1
        append_jsonl(
            args.out,
            {
                "id": f"cand-{counter:04d}",
                "question": question,
                "gold_chunk_ids": paragraph.chunk_ids,
                "paragraph_id": paragraph.paragraph_id,
                "status": "потребує ручної перевірки",
            },
        )
        written += 1

    print(f"Записано кандидатів: {written} у {args.out}")
    print("Перевір кожне питання вручну і перенеси прийнятні рядки в data/eval.jsonl.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
