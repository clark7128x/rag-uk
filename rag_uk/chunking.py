from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Iterable

from .config import ChunkingConfig

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE_GAP = re.compile(r"(?<=[.!?…])\s+")

# Скорочення, після яких крапка не закінчує речення.
_ABBREV = {
    "ст", "пп", "п", "т", "грн", "коп", "р", "м", "вул", "просп", "напр",
    "див", "табл", "рис", "абз", "ч", "ред", "ім", "проф", "акад",
    "млн", "млрд", "тис", "т.д", "т.п", "т.ін", "ін",
}


@dataclass
class Document:
    doc_id: str
    text: str


def paragraph_id(doc_id: str, index: int) -> str:
    return f"{doc_id}:p{index:03d}"


@dataclass
class Paragraph:
    paragraph_id: str
    doc_id: str
    index: int
    section: str
    text: str
    char_start: int
    char_end: int
    chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Paragraph":
        return cls(**data)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    ordinal: int
    section: str
    text: str
    char_start: int
    char_end: int
    paragraph_start: int
    paragraph_end: int

    def index_text(self, prefix_section: bool) -> str:
        if prefix_section and self.section:
            return f"{self.section}\n{self.text}"
        return self.text

    def paragraph_ids(self) -> list[str]:
        return [
            paragraph_id(self.doc_id, i)
            for i in range(self.paragraph_start, self.paragraph_end + 1)
        ]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


def split_paragraphs(doc_id: str, text: str) -> list[Paragraph]:
    paragraphs: list[Paragraph] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    buf_start = 0
    buf_end = 0

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        index = len(paragraphs)
        paragraphs.append(
            Paragraph(
                paragraph_id=paragraph_id(doc_id, index),
                doc_id=doc_id,
                index=index,
                section=" > ".join(title for _, title in stack),
                text="\n".join(buf),
                char_start=buf_start,
                char_end=buf_end,
            )
        )
        buf = []

    pos = 0
    for raw in text.splitlines(keepends=True):
        line = raw.rstrip("\r\n")
        stripped = line.strip()
        heading = _HEADING.match(stripped)
        if heading:
            flush()
            level = len(heading.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading.group(2).strip()))
        elif not stripped:
            flush()
        else:
            if not buf:
                buf_start = pos
            buf.append(line)
            buf_end = pos + len(line)
        pos += len(raw)
    flush()
    return paragraphs


def _false_boundary(text: str, end: int) -> bool:
    tail = text[:end].rstrip()
    if not tail:
        return True
    token_start = max(tail.rfind(" "), tail.rfind("\n")) + 1
    word = tail[token_start:].rstrip(".!?…")
    if not word:
        return True
    if word.lower() in _ABBREV:
        return True
    if len(word) == 1 and word.isupper():
        return True
    if word.isdigit() and (token_start == 0 or tail[token_start - 1] == "\n"):
        return True
    rest = text[end:].lstrip()
    return bool(rest) and rest[0].islower()


def split_sentences(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_GAP.finditer(text):
        if _false_boundary(text, match.start()):
            continue
        spans.append((start, match.start()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))

    trimmed: list[tuple[int, int]] = []
    for a, b in spans:
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if b > a:
            trimmed.append((a, b))
    return trimmed


def _hard_split(text: str, start: int, end: int, limit: int) -> list[tuple[int, int]]:
    pieces: list[tuple[int, int]] = []
    pos = start
    while end - pos > limit:
        cut = text[pos : pos + limit].rfind(" ")
        if cut <= 0:
            cut = limit  # суцільне слово довше за ліміт ріжемо примусово
        pieces.append((pos, pos + cut))
        pos += cut
        while pos < end and text[pos] == " ":
            pos += 1
    if pos < end:
        pieces.append((pos, end))
    return pieces


@dataclass(frozen=True)
class _Unit:
    start: int
    end: int
    paragraph: int
    section: str


def _units(text: str, paragraphs: list[Paragraph], cfg: ChunkingConfig) -> list[_Unit]:
    units: list[_Unit] = []
    for paragraph in paragraphs:
        for a, b in split_sentences(paragraph.text):
            start, end = paragraph.char_start + a, paragraph.char_start + b
            if end - start > cfg.hard_max_chars:
                for s, e in _hard_split(text, start, end, cfg.hard_max_chars):
                    units.append(_Unit(s, e, paragraph.index, paragraph.section))
            else:
                units.append(_Unit(start, end, paragraph.index, paragraph.section))
    return units


def _pack(doc: Document, units: list[_Unit], cfg: ChunkingConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    i = 0
    while i < len(units):
        j = i
        size = 0
        while j < len(units):
            length = units[j].end - units[j].start
            gap = units[j].start - units[j - 1].end if j > i else 0
            if j > i:
                # Чанк не перетинає межу розділу: інакше заголовок потрапляє в текст,
                # а мітка розділу описує лише початок чанка.
                if units[j].section != units[j - 1].section:
                    break
                if size + gap + length > cfg.target_chars:
                    break
                # Новий абзац починаємо з нового чанка, якщо поточний уже не порожній.
                if units[j].paragraph != units[j - 1].paragraph and size >= cfg.min_chars:
                    break
            size += gap + length
            j += 1

        start, end = units[i].start, units[j - 1].end
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:{len(chunks):04d}",
                doc_id=doc.doc_id,
                ordinal=len(chunks),
                section=units[i].section,
                text=doc.text[start:end],
                char_start=start,
                char_end=end,
                paragraph_start=units[i].paragraph,
                paragraph_end=units[j - 1].paragraph,
            )
        )
        if j >= len(units):
            break

        # Через межу розділу перекриття не береться.
        if units[j].section != units[j - 1].section:
            i = j
            continue

        # Перекриття беремо цілими реченнями з хвоста щойно закритого чанка.
        t = j
        while t - 1 > i and end - units[t - 1].start <= cfg.overlap_chars:
            t -= 1
        i = t
    return chunks


def chunk_document(doc: Document, cfg: ChunkingConfig) -> tuple[list[Chunk], list[Paragraph]]:
    paragraphs = split_paragraphs(doc.doc_id, doc.text)
    chunks = _pack(doc, _units(doc.text, paragraphs, cfg), cfg)
    for paragraph in paragraphs:
        paragraph.chunk_ids = [
            chunk.chunk_id
            for chunk in chunks
            if chunk.paragraph_start <= paragraph.index <= chunk.paragraph_end
        ]
    return chunks, paragraphs


def chunk_corpus(
    docs: Iterable[Document], cfg: ChunkingConfig
) -> tuple[list[Chunk], list[Paragraph]]:
    chunks: list[Chunk] = []
    paragraphs: list[Paragraph] = []
    for doc in docs:
        doc_chunks, doc_paragraphs = chunk_document(doc, cfg)
        chunks.extend(doc_chunks)
        paragraphs.extend(doc_paragraphs)
    return chunks, paragraphs
