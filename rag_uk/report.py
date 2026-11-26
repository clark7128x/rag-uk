from __future__ import annotations

MISSING = "не виміряно"


def _fmt(value, spec: str = "{:.3f}") -> str:
    return MISSING if value is None else spec.format(value)


def _ms(value) -> str:
    return MISSING if value is None else f"{value * 1000:.0f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def render_markdown(payload: dict) -> str:
    config = payload.get("config", {})
    k_values = config.get("eval", {}).get("k_values", [1, 3, 5])
    dataset = payload.get("dataset", {})
    lines = [
        "# Результати оцінки",
        "",
        f"Набір: `{dataset.get('path', '—')}`. Рядків: {dataset.get('rows', 0)}. "
        f"Пропущено: {dataset.get('skipped', 0)}.",
        "",
        "## Пошук",
        "",
    ]

    rows = []
    for variant in payload.get("retrieval", []):
        metrics = variant.get("metrics") or {}
        latency = variant.get("latency_s") or {}
        rows.append(
            [variant.get("mode", "—"), variant.get("embedding_model") or "—"]
            + [_fmt(metrics.get(f"recall@{k}")) for k in k_values]
            + [_fmt(metrics.get("mrr")), _ms(latency.get("p50"))]
        )
    lines += _table(
        ["Режим", "Ембединги"] + [f"recall@{k}" for k in k_values] + ["MRR", "p50 пошуку, мс"],
        rows,
    )

    errors = [
        f"- {v.get('mode')} / {v.get('embedding_model') or '—'}: {v['error']}"
        for v in payload.get("retrieval", [])
        if v.get("error")
    ]
    if errors:
        lines += ["", "Помилки запуску:", *errors]

    generation = payload.get("generation") or {}
    metrics = generation.get("metrics") or {}
    latency = generation.get("latency_s") or {}
    lines += ["", "## Генерація", ""]
    lines += _table(
        ["Режим", "Відповідей", "Усі цитати в контексті", "Частка", "p50 генерації, мс", "Вартість"],
        [
            [
                generation.get("mode", "—"),
                str(metrics.get("answers", 0)) if metrics else MISSING,
                _fmt(metrics.get("all_citations_in_context"), "{:.0f}"),
                _fmt(metrics.get("grounding_rate")),
                _ms(latency.get("p50")),
                _fmt(generation.get("cost"), "{:.4f}"),
            ]
        ],
    )
    if generation.get("skipped_reason"):
        lines += ["", f"Пропущено: {generation['skipped_reason']}."]
    if generation.get("failed"):
        lines += ["", f"Викликів моделі з помилкою: {generation['failed']}."]

    calls = (payload.get("calls") or {}).get("by_model", [])
    lines += ["", "## Виклики моделей", ""]
    lines += _table(
        ["Тип", "Модель", "Викликів", "Вхідні токени", "Вихідні токени", "p50, мс", "p95, мс", "Вартість"],
        [
            [
                row["kind"],
                row["model"],
                str(row["calls"]),
                _fmt(row["input_tokens"], "{:.0f}"),
                _fmt(row["output_tokens"], "{:.0f}"),
                _ms(row["latency_p50_s"]),
                _ms(row["latency_p95_s"]),
                _fmt(row["cost"], "{:.4f}"),
            ]
            for row in calls
        ],
    )
    return "\n".join(lines) + "\n"
