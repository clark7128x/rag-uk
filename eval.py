#!/usr/bin/env python3
"""Оцінка режимів пошуку та генерації на JSONL-наборі."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from rag_uk.calllog import CallLog
from rag_uk.config import VECTOR_MODES, ConfigError, load_config
from rag_uk.corpus import ensure_chunks
from rag_uk.generate import answer_question
from rag_uk.jsonio import read_jsonl, write_jsonl
from rag_uk.llm import LLMClient, LLMError
from rag_uk.metrics import citation_summary, latency_stats, retrieval_summary
from rag_uk.report import render_markdown
from rag_uk.retrieve import Retriever


def load_rows(path: Path, known_ids: set[str]) -> tuple[list[dict], list[str]]:
    if not path.exists():
        return [], [f"немає файлу {path}"]
    rows: list[dict] = []
    problems: list[str] = []
    for number, raw in enumerate(read_jsonl(path), start=1):
        qid, question, gold = raw.get("id"), raw.get("question"), raw.get("gold_chunk_ids")
        if not isinstance(qid, str) or not qid:
            problems.append(f"рядок {number}: немає id")
            continue
        if not isinstance(question, str) or not question.strip():
            problems.append(f"рядок {number}: порожнє питання")
            continue
        if not isinstance(gold, list) or not gold:
            problems.append(f"рядок {number}: порожні gold_chunk_ids")
            continue
        unknown = [g for g in gold if g not in known_ids]
        if unknown:
            problems.append(f"рядок {number}: невідомі gold_chunk_ids {unknown}")
            continue
        rows.append({"id": qid, "question": question.strip(), "gold_chunk_ids": gold})
    return rows, problems


def plan_variants(cfg, modes: list[str]) -> list[tuple[str, str | None]]:
    variants = []
    for mode in modes:
        if mode in VECTOR_MODES:
            variants.extend((mode, name) for name in cfg.eval.embedding_models)
        else:
            variants.append((mode, None))
    return variants


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--modes", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-generation", action="store_true")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        chunks, _, meta = ensure_chunks(cfg)
        modes = args.modes or list(cfg.eval.modes)
        rows, problems = load_rows(cfg.eval.dataset, {c.chunk_id for c in chunks})
    except (ConfigError, ValueError) as exc:
        print(f"Вхідні дані: {exc}", file=sys.stderr)
        return 2

    for problem in problems:
        print(f"Пропущено: {problem}", file=sys.stderr)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        print(
            "Набір порожній. Запусти tools/label_cli.py і повтори eval.py. "
            "Метрики не рахуються на вигаданих даних.",
            file=sys.stderr,
        )
        return 2

    call_log = CallLog(
        cfg.eval.call_log, cfg.llm.price_per_1m_input, cfg.llm.price_per_1m_output
    )
    details: list[dict] = []
    retrieval_results: list[dict] = []

    for mode, embedding_model in plan_variants(cfg, modes):
        variant = {"mode": mode, "embedding_model": embedding_model, "error": None}
        try:
            retriever = Retriever(
                chunks, cfg, call_log=call_log, embedding_model=embedding_model
            )
            per_row, latencies = [], []
            for row in rows:
                started = perf_counter()
                hits = retriever.search(row["question"], mode=mode)
                latencies.append(perf_counter() - started)
                retrieved = [hit.chunk.chunk_id for hit in hits]
                per_row.append({"retrieved": retrieved, "gold_chunk_ids": row["gold_chunk_ids"]})
                details.append(
                    {
                        "kind": "retrieval",
                        "id": row["id"],
                        "mode": mode,
                        "embedding_model": embedding_model,
                        "retrieved": retrieved,
                        "gold_chunk_ids": row["gold_chunk_ids"],
                    }
                )
            variant["metrics"] = retrieval_summary(per_row, cfg.eval.k_values)
            variant["latency_s"] = latency_stats(latencies)
        except Exception as exc:  # один режим не має зупиняти всю оцінку
            variant["error"] = f"{type(exc).__name__}: {exc}"
            variant["metrics"] = None
            variant["latency_s"] = None
        retrieval_results.append(variant)

    generation = {
        "mode": cfg.retrieval.mode,
        "embedding_model": (
            cfg.retrieval.vector.model if cfg.retrieval.mode in VECTOR_MODES else None
        ),
        "metrics": None,
        "latency_s": None,
        "cost": None,
        "failed": 0,
        "skipped_reason": None,
    }
    if args.no_generation or not cfg.eval.generation:
        generation["skipped_reason"] = "генерацію вимкнено"
    elif not cfg.llm.configured:
        generation["skipped_reason"] = "не задано LLM_API_BASE або LLM_MODEL"
    else:
        retriever = Retriever(chunks, cfg, call_log=call_log)
        llm = LLMClient(cfg.llm, call_log)
        reports, latencies = [], []
        chat_start = len(call_log.records)
        for row in rows:
            try:
                hits = retriever.search(row["question"], mode=cfg.retrieval.mode)
                answer = answer_question(row["question"], hits, llm)
            except (LLMError, RuntimeError, ValueError) as exc:
                generation["failed"] += 1
                details.append(
                    {"kind": "generation", "id": row["id"], "error": str(exc)}
                )
                continue
            reports.append(
                {
                    "has_citations": answer.has_citations,
                    "all_in_context": answer.all_in_context,
                    "unknown_citations": answer.unknown_citations,
                }
            )
            if answer.latency_s is not None:
                latencies.append(answer.latency_s)
            details.append(
                {
                    "kind": "generation",
                    "id": row["id"],
                    "answer": answer.text,
                    "context_ids": answer.context_ids,
                    "citations": answer.citations,
                    "unknown_citations": answer.unknown_citations,
                    "all_in_context": answer.all_in_context,
                    "skipped_reason": answer.skipped_reason,
                }
            )
        generation["metrics"] = citation_summary(reports)
        generation["latency_s"] = latency_stats(latencies)
        chat_calls = CallLog(
            None, cfg.llm.price_per_1m_input, cfg.llm.price_per_1m_output
        )
        chat_calls.records = [
            r for r in call_log.records[chat_start:] if r["kind"] == "chat"
        ]
        generation["cost"] = chat_calls.summary()["cost_total"]

    payload = {
        "run_id": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "chunking": asdict(cfg.chunking),
            "retrieval": {
                "mode": cfg.retrieval.mode,
                "top_k": cfg.retrieval.top_k,
                "candidate_k": cfg.retrieval.candidate_k,
                "rrf_k": cfg.retrieval.rrf_k,
                "bm25": asdict(cfg.retrieval.bm25),
                "reranker": asdict(cfg.retrieval.reranker),
            },
            "eval": {"k_values": list(cfg.eval.k_values)},
            "llm_model": cfg.llm.model,
        },
        "corpus": meta,
        "dataset": {
            "path": str(cfg.eval.dataset),
            "rows": len(rows),
            "skipped": len(problems),
            "problems": problems,
        },
        "retrieval": retrieval_results,
        "generation": generation,
        "calls": call_log.summary(),
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_dir = cfg.eval.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"eval_{stamp}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(results_dir / f"details_{stamp}.jsonl", details)
    markdown = render_markdown(payload)
    (results_dir / "metrics.md").write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
