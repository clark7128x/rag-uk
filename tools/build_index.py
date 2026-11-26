#!/usr/bin/env python3
"""Нарізка корпусу на чанки та підготовка кешу ембедингів."""
from __future__ import annotations

import argparse
import sys

from rag_uk.config import ConfigError, load_config
from rag_uk.corpus import ensure_chunks
from rag_uk.retrieve import Retriever


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--embeddings", choices=["none", "configured", "all"], default="configured")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        chunks, paragraphs, meta = ensure_chunks(cfg, rebuild=args.rebuild)
    except ConfigError as exc:
        print(f"Конфіг: {exc}", file=sys.stderr)
        return 2

    print(
        f"Документів: {meta['documents']}. Абзаців: {meta['paragraphs']}. "
        f"Чанків: {len(chunks)}. Кеш: {cfg.cache_dir}"
    )
    if args.embeddings == "none":
        return 0

    names = (
        list(cfg.retrieval.vector.profiles)
        if args.embeddings == "all"
        else [cfg.retrieval.vector.model]
    )
    for name in names:
        retriever = Retriever(chunks, cfg, embedding_model=name)
        try:
            retriever._vector_index()
        except (RuntimeError, OSError) as exc:
            print(f"{name}: {exc}", file=sys.stderr)
            return 1
        print(f"Ембединги готові: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
