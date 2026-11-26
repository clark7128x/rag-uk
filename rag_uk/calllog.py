from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .jsonio import append_jsonl
from .metrics import percentile


class CallLog:
    def __init__(
        self,
        path: str | Path | None = None,
        price_per_1m_input: float | None = None,
        price_per_1m_output: float | None = None,
    ):
        self.path = Path(path) if path is not None else None
        self.price_in = price_per_1m_input
        self.price_out = price_per_1m_output
        self.records: list[dict[str, Any]] = []

    def record(
        self,
        *,
        kind: str,
        model: str,
        input_tokens: int | None,
        output_tokens: int | None,
        latency_s: float,
        ok: bool = True,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "kind": kind,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_s": latency_s,
            "ok": ok,
            "error": error,
        }
        if extra:
            row.update(extra)
        self.records.append(row)
        if self.path is not None:
            append_jsonl(self.path, row)

    def _cost(self, records: list[dict[str, Any]]) -> float | None:
        if self.price_in is None or self.price_out is None or not records:
            return None
        total = 0.0
        for row in records:
            if row["input_tokens"] is None or row["output_tokens"] is None:
                return None
            total += row["input_tokens"] / 1_000_000 * self.price_in
            total += row["output_tokens"] / 1_000_000 * self.price_out
        return total

    @staticmethod
    def _sum_tokens(records: list[dict[str, Any]], field: str) -> int | None:
        values = [r[field] for r in records if r[field] is not None]
        return sum(values) if values else None

    def summary(self) -> dict[str, Any]:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in self.records:
            groups.setdefault((row["kind"], row["model"]), []).append(row)

        by_model = []
        for (kind, model), rows in sorted(groups.items()):
            latencies = [r["latency_s"] for r in rows]
            by_model.append(
                {
                    "kind": kind,
                    "model": model,
                    "calls": len(rows),
                    "failed": sum(1 for r in rows if not r["ok"]),
                    "input_tokens": self._sum_tokens(rows, "input_tokens"),
                    "output_tokens": self._sum_tokens(rows, "output_tokens"),
                    "latency_total_s": sum(latencies),
                    "latency_p50_s": percentile(latencies, 50),
                    "latency_p95_s": percentile(latencies, 95),
                    "cost": self._cost(rows) if kind == "chat" else None,
                }
            )
        return {
            "calls": len(self.records),
            "failed": sum(1 for r in self.records if not r["ok"]),
            "cost_total": self._cost([r for r in self.records if r["kind"] == "chat"]),
            "by_model": by_model,
        }
