from __future__ import annotations

from dataclasses import dataclass

from .citations import citation_report
from .llm import LLMClient
from .retrieve import Hit

SYSTEM_PROMPT = (
    "Ти відповідаєш українською мовою лише на основі наданих витягів з документів.\n"
    "Кожне твердження підтверджуй дослівною цитатою в лапках «...».\n"
    "Одразу після цитати став ідентифікатор чанка у квадратних дужках, наприклад [example:0003].\n"
    "Використовуй тільки ідентифікатори з контексту.\n"
    "Якщо у контексті є виняток або умова «якщо», «крім», «за винятком», що стосується питання, "
    "наводь їх разом із загальним правилом.\n"
    "Якщо контекст не містить відповіді, напиши «Недостатньо даних у корпусі» і не додавай цитат."
)

NO_CONTEXT_ANSWER = "Недостатньо даних у корпусі."


@dataclass
class AnswerResult:
    text: str
    context_ids: list[str]
    citations: list[str]
    unknown_citations: list[str]
    has_citations: bool
    all_in_context: bool
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_s: float | None = None
    skipped_reason: str | None = None


def build_messages(question: str, hits: list[Hit]) -> list[dict[str, str]]:
    blocks = []
    for hit in hits:
        chunk = hit.chunk
        header = (
            f"[{chunk.chunk_id}] документ={chunk.doc_id} "
            f"розділ={chunk.section or '—'} абзаци={','.join(chunk.paragraph_ids())}"
        )
        blocks.append(f"{header}\n{chunk.text}")
    context = "\n\n".join(blocks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Контекст:\n{context}\n\nПитання: {question}\n\nВідповідь:"},
    ]


def strip_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def answer_question(question: str, hits: list[Hit], llm: LLMClient) -> AnswerResult:
    if not question or not question.strip():
        raise ValueError("Порожнє питання")
    if not hits:
        return AnswerResult(
            text=NO_CONTEXT_ANSWER,
            context_ids=[],
            citations=[],
            unknown_citations=[],
            has_citations=False,
            all_in_context=False,
            skipped_reason="no_context",
        )

    result = llm.chat(build_messages(question, hits))
    text = strip_fences(result.text)
    context_ids = [hit.chunk.chunk_id for hit in hits]
    report = citation_report(text, context_ids)
    return AnswerResult(
        text=text,
        context_ids=context_ids,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_s=result.latency_s,
        **report,
    )
