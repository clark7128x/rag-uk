#!/usr/bin/env python3
"""Ручна розмітка: показує абзац і питає питання, на яке він відповідає."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_uk.config import ConfigError, load_config
from rag_uk.corpus import ensure_chunks
from rag_uk.jsonio import append_jsonl, read_jsonl
from rag_uk.labeling import LabelError, make_record, next_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--doc", help="показувати лише цей doc_id")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dataset", help="шлях замість eval.dataset")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        chunks, paragraphs, _ = ensure_chunks(cfg)
    except ConfigError as exc:
        print(f"Конфіг: {exc}", file=sys.stderr)
        return 2

    dataset = cfg.eval.dataset if args.dataset is None else Path(args.dataset)
    existing = read_jsonl(dataset) if dataset.exists() else []
    existing_ids = {row["id"] for row in existing}
    existing_questions = {row["question"].strip().casefold() for row in existing}
    known_ids = {chunk.chunk_id for chunk in chunks}

    if args.doc:
        paragraphs = [p for p in paragraphs if p.doc_id == args.doc]
    selected = paragraphs[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]

    print("Enter пропускає абзац, q завершує. Порожній набір не заповнюється автоматично.")
    added = 0
    try:
        for paragraph in selected:
            print("\n" + "=" * 72)
            print(f"{paragraph.paragraph_id} | розділ: {paragraph.section or '—'}")
            print(f"чанки: {', '.join(paragraph.chunk_ids) or '—'}")
            print("-" * 72)
            print(paragraph.text)
            question = input("\nПитання: ").strip()
            if question.lower() in {"q", "quit", "exit"}:
                break
            if not question:
                continue
            default_gold = ",".join(paragraph.chunk_ids)
            raw = input(f"gold_chunk_ids [Enter = {default_gold}]: ").strip()
            gold = [g.strip() for g in raw.split(",") if g.strip()] if raw else paragraph.chunk_ids
            try:
                record = make_record(
                    question, gold, known_ids, existing_questions, next_id(existing_ids)
                )
            except LabelError as exc:
                print(f"Пропущено: {exc}")
                continue
            append_jsonl(dataset, record)
            existing_ids.add(record["id"])
            existing_questions.add(record["question"].casefold())
            added += 1
            print(f"Записано {record['id']}")
    except (KeyboardInterrupt, EOFError):
        print("\nПерервано.")

    print(f"\nДодано: {added}. Усього в {dataset}: {len(existing_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
