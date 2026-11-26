#!/usr/bin/env python3
"""Одне питання до системи: пошук і відповідь із цитатами."""
from __future__ import annotations

import argparse
import sys

from rag_uk.calllog import CallLog
from rag_uk.config import ConfigError, load_config
from rag_uk.corpus import ensure_chunks
from rag_uk.generate import answer_question
from rag_uk.llm import LLMClient, LLMError
from rag_uk.retrieve import Retriever


def print_hits(hits) -> None:
    for hit in hits:
        chunk = hit.chunk
        print(f"\n[{chunk.chunk_id}] {hit.source} score={hit.score:.4f}")
        print(f"розділ: {chunk.section or '—'} | абзаци: {', '.join(chunk.paragraph_ids())}")
        print(chunk.text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--config")
    parser.add_argument("--mode")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--embedding-model")
    parser.add_argument("--no-generation", action="store_true")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        chunks, _, _ = ensure_chunks(cfg)
    except ConfigError as exc:
        print(f"Конфіг: {exc}", file=sys.stderr)
        return 2

    call_log = CallLog(
        cfg.cache_dir / "ask_calls.jsonl",
        cfg.llm.price_per_1m_input,
        cfg.llm.price_per_1m_output,
    )
    retriever = Retriever(
        chunks, cfg, call_log=call_log, embedding_model=args.embedding_model
    )
    try:
        hits = retriever.search(args.question, mode=args.mode, top_k=args.top_k)
    except (ValueError, RuntimeError) as exc:
        print(f"Пошук: {exc}", file=sys.stderr)
        return 1

    if not hits:
        print("Нічого не знайдено.")
        return 1

    if args.no_generation or not cfg.llm.configured:
        if not cfg.llm.configured and not args.no_generation:
            print("LLM не налаштовано, показую лише знайдені чанки.\n")
        print_hits(hits)
        return 0

    try:
        answer = answer_question(args.question, hits, LLMClient(cfg.llm, call_log))
    except LLMError as exc:
        print(f"Генерація: {exc}", file=sys.stderr)
        print_hits(hits)
        return 1

    print(answer.text)
    print("\nДжерела:")
    for hit in hits:
        chunk = hit.chunk
        mark = "використано" if chunk.chunk_id in answer.citations else "у контексті"
        print(f"[{chunk.chunk_id}] {mark} | {chunk.section or '—'} | {', '.join(chunk.paragraph_ids())}")
    if answer.unknown_citations:
        print(f"\nУвага: цитати поза контекстом: {answer.unknown_citations}")

    summary = call_log.summary()
    print(f"\nВикликів: {summary['calls']}. Вартість: {summary['cost_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
