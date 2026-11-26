from __future__ import annotations

from time import perf_counter

from .calllog import CallLog
from .chunking import Chunk
from .config import RerankerConfig


class CrossEncoderReranker:
    def __init__(self, cfg: RerankerConfig, call_log: CallLog | None = None):
        self.cfg = cfg
        self.call_log = call_log
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "Реранкер потребує sentence-transformers: pip install -e '.[vector]'"
                ) from exc
            self._model = CrossEncoder(self.cfg.model, device=self.cfg.device)
        return self._model

    def _count_tokens(self, model, pairs: list[list[str]]) -> int | None:
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            return None
        try:
            encoded = tokenizer(
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                truncation=True,
                max_length=getattr(model, "max_seq_length", None),
            )
            return sum(len(ids) for ids in encoded["input_ids"])
        except Exception:
            return None

    def rerank(
        self, query: str, chunks: list[Chunk], top_n: int, prefix_section: bool = False
    ) -> list[tuple[str, float]]:
        if not chunks or top_n <= 0:
            return []
        model = self._load()
        pairs = [[query, c.index_text(prefix_section)] for c in chunks]
        started = perf_counter()
        scores = model.predict(
            pairs, batch_size=self.cfg.batch_size, show_progress_bar=False
        )
        latency = perf_counter() - started
        if self.call_log:
            self.call_log.record(
                kind="rerank",
                model=self.cfg.model,
                input_tokens=self._count_tokens(model, pairs),
                output_tokens=0,
                latency_s=latency,
                extra={"pairs": len(pairs)},
            )
        ordered = sorted(
            zip([c.chunk_id for c in chunks], (float(s) for s in scores)),
            key=lambda kv: (-kv[1], kv[0]),
        )
        return ordered[:top_n]
