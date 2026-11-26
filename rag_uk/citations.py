from __future__ import annotations

import re

_BRACKETS = re.compile(r"\[([^\[\]\n]{1,200})\]")
_CHUNK_ID = re.compile(r"^[\w.\-]+:\d{4}$", re.UNICODE)


def extract_citations(text: str) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("text має бути рядком")
    found: list[str] = []
    for match in _BRACKETS.finditer(text):
        candidate = match.group(1).strip()
        if _CHUNK_ID.match(candidate) and candidate not in found:
            found.append(candidate)
    return found


def citation_report(text: str, context_ids: list[str]) -> dict:
    citations = extract_citations(text)
    known = set(context_ids)
    unknown = [c for c in citations if c not in known]
    return {
        "citations": citations,
        "unknown_citations": unknown,
        "has_citations": bool(citations),
        # Відповідь без цитат не зараховується як обґрунтована.
        "all_in_context": bool(citations) and not unknown,
    }
