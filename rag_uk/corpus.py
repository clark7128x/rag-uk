from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

from .chunking import Chunk, Document, Paragraph, chunk_corpus
from .config import Config, ConfigError
from .jsonio import read_jsonl, write_jsonl


def load_documents(cfg: Config) -> list[Document]:
    if not cfg.corpus_dir.is_dir():
        raise ConfigError(f"Немає каталогу корпусу: {cfg.corpus_dir}")
    paths: set[Path] = set()
    for pattern in cfg.corpus_glob:
        paths.update(p for p in cfg.corpus_dir.glob(pattern) if p.is_file())
    if not paths:
        raise ConfigError(
            f"У {cfg.corpus_dir} немає файлів за масками {list(cfg.corpus_glob)}"
        )

    docs: list[Document] = []
    seen: dict[str, Path] = {}
    for path in sorted(paths):
        doc_id = re.sub(r"[^\w.\-]", "_", path.stem, flags=re.UNICODE)
        if doc_id in seen:
            raise ConfigError(f"Конфлікт doc_id {doc_id}: {seen[doc_id]} і {path}")
        seen[doc_id] = path
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        docs.append(Document(doc_id=doc_id, text=text))
    return docs


def corpus_fingerprint(docs: list[Document], cfg: Config) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(asdict(cfg.chunking), sort_keys=True).encode("utf-8"))
    for doc in docs:
        digest.update(b"\x00")
        digest.update(doc.doc_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(doc.text.encode("utf-8"))
    return digest.hexdigest()


def ensure_chunks(
    cfg: Config, rebuild: bool = False
) -> tuple[list[Chunk], list[Paragraph], dict]:
    chunks_path = cfg.cache_dir / "chunks.jsonl"
    paragraphs_path = cfg.cache_dir / "paragraphs.jsonl"
    meta_path = cfg.cache_dir / "meta.json"

    if not rebuild and chunks_path.exists() and paragraphs_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        docs = load_documents(cfg)
        if meta.get("fingerprint") == corpus_fingerprint(docs, cfg):
            chunks = [Chunk.from_dict(r) for r in read_jsonl(chunks_path)]
            paragraphs = [Paragraph.from_dict(r) for r in read_jsonl(paragraphs_path)]
            return chunks, paragraphs, meta

    docs = load_documents(cfg)
    chunks, paragraphs = chunk_corpus(docs, cfg.chunking)
    meta = {
        "fingerprint": corpus_fingerprint(docs, cfg),
        "documents": len(docs),
        "chunks": len(chunks),
        "paragraphs": len(paragraphs),
        "chunking": asdict(cfg.chunking),
    }
    write_jsonl(chunks_path, [c.to_dict() for c in chunks])
    write_jsonl(paragraphs_path, [p.to_dict() for p in paragraphs])
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return chunks, paragraphs, meta
