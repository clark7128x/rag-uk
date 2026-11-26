from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

RETRIEVAL_MODES = ("bm25", "vector", "hybrid", "hybrid_rerank")
VECTOR_MODES = ("vector", "hybrid", "hybrid_rerank")

DEFAULT_PROFILES = {
    "e5-base": {
        "repo": "intfloat/multilingual-e5-base",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "minilm-l12": {
        "repo": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "query_prefix": "",
        "passage_prefix": "",
    },
}


class ConfigError(ValueError):
    pass


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


@dataclass(frozen=True)
class ChunkingConfig:
    target_chars: int = 900
    overlap_chars: int = 150
    min_chars: int = 120
    hard_max_chars: int = 1400
    prefix_section: bool = True

    def __post_init__(self) -> None:
        _need(self.target_chars > 0, "chunking.target_chars має бути > 0")
        _need(
            self.hard_max_chars >= self.target_chars,
            "chunking.hard_max_chars має бути >= target_chars",
        )
        _need(
            0 <= self.overlap_chars < self.target_chars,
            "chunking.overlap_chars має бути в [0, target_chars)",
        )
        _need(
            0 <= self.min_chars <= self.target_chars,
            "chunking.min_chars має бути в [0, target_chars]",
        )


@dataclass(frozen=True)
class EmbeddingProfile:
    name: str
    repo: str
    query_prefix: str = ""
    passage_prefix: str = ""


@dataclass(frozen=True)
class VectorConfig:
    model: str = "e5-base"
    profiles: dict[str, EmbeddingProfile] = None  # type: ignore[assignment]
    batch_size: int = 32
    device: str = "cpu"

    def profile(self, name: str | None = None) -> EmbeddingProfile:
        key = name or self.model
        if key not in self.profiles:
            raise ConfigError(
                f"Невідомий профіль ембедингів: {key}. Доступні: {sorted(self.profiles)}"
            )
        return self.profiles[key]


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        _need(self.k1 > 0, "retrieval.bm25.k1 має бути > 0")
        _need(0 <= self.b <= 1, "retrieval.bm25.b має бути в [0, 1]")


@dataclass(frozen=True)
class RerankerConfig:
    model: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 8
    device: str = "cpu"


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str = "hybrid"
    top_k: int = 5
    candidate_k: int = 30
    rrf_k: int = 60
    bm25: BM25Config = BM25Config()
    vector: VectorConfig = None  # type: ignore[assignment]
    reranker: RerankerConfig = RerankerConfig()

    def __post_init__(self) -> None:
        _need(
            self.mode in RETRIEVAL_MODES,
            f"retrieval.mode має бути одним з {RETRIEVAL_MODES}, отримано {self.mode!r}",
        )
        _need(self.top_k >= 1, "retrieval.top_k має бути >= 1")
        _need(
            self.candidate_k >= self.top_k,
            "retrieval.candidate_k має бути >= top_k",
        )
        _need(self.rrf_k >= 1, "retrieval.rrf_k має бути >= 1")


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = ""
    api_key: str | None = None
    model: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 700
    timeout_s: float = 60.0
    max_retries: int = 3
    retry_base_delay_s: float = 1.0
    retry_max_delay_s: float = 20.0
    price_per_1m_input: float | None = None
    price_per_1m_output: float | None = None

    def __post_init__(self) -> None:
        _need(self.temperature >= 0, "llm.temperature має бути >= 0")
        _need(self.max_output_tokens > 0, "llm.max_output_tokens має бути > 0")
        _need(self.timeout_s > 0, "llm.timeout_s має бути > 0")
        _need(self.max_retries >= 0, "llm.max_retries має бути >= 0")
        for name in ("price_per_1m_input", "price_per_1m_output"):
            value = getattr(self, name)
            _need(value is None or value >= 0, f"llm.{name} має бути >= 0 або null")

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model)


@dataclass(frozen=True)
class EvalConfig:
    dataset: Path
    k_values: tuple[int, ...] = (1, 3, 5)
    modes: tuple[str, ...] = RETRIEVAL_MODES
    embedding_models: tuple[str, ...] = ("e5-base", "minilm-l12")
    generation: bool = True
    results_dir: Path = Path("results")
    call_log: Path = Path("results/calls.jsonl")


@dataclass(frozen=True)
class Config:
    corpus_dir: Path
    corpus_glob: tuple[str, ...]
    cache_dir: Path
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    eval: EvalConfig
    raw: dict[str, Any]


def _build(cls, data: dict, section: str):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    _need(not unknown, f"{section}: невідомі ключі {sorted(unknown)}")
    try:
        return cls(**data)
    except TypeError as exc:
        raise ConfigError(f"{section}: {exc}") from exc


def load_config(path: str | Path | None = None, env: Mapping[str, str] | None = None) -> Config:
    env = os.environ if env is None else env
    cfg_path = Path(path or env.get("RAG_CONFIG") or "config.yaml")
    if not cfg_path.is_file():
        raise ConfigError(f"Немає конфігу: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    _need(isinstance(data, dict), "config.yaml має містити словник")
    base = cfg_path.resolve().parent

    def resolve(value: str | Path) -> Path:
        return (base / value).resolve()

    if env.get("RAG_RETRIEVAL_MODE"):
        data.setdefault("retrieval", {})["mode"] = env["RAG_RETRIEVAL_MODE"]
    if env.get("RAG_EMBEDDING_MODEL"):
        data.setdefault("retrieval", {}).setdefault("vector", {})["model"] = env[
            "RAG_EMBEDDING_MODEL"
        ]

    corpus = data.get("corpus") or {}
    index = data.get("index") or {}
    retrieval = dict(data.get("retrieval") or {})
    llm_raw = dict(data.get("llm") or {})
    eval_raw = dict(data.get("eval") or {})

    chunking = _build(ChunkingConfig, data.get("chunking") or {}, "chunking")
    bm25 = _build(BM25Config, retrieval.get("bm25") or {}, "retrieval.bm25")
    reranker = _build(RerankerConfig, retrieval.get("reranker") or {}, "retrieval.reranker")

    vector_raw = dict(retrieval.get("vector") or {})
    profiles_raw = vector_raw.pop("models", None) or DEFAULT_PROFILES
    profiles = {
        name: EmbeddingProfile(
            name=name,
            repo=spec["repo"],
            query_prefix=spec.get("query_prefix", ""),
            passage_prefix=spec.get("passage_prefix", ""),
        )
        for name, spec in profiles_raw.items()
    }
    vector = _build(VectorConfig, {**vector_raw, "profiles": profiles}, "retrieval.vector")
    vector.profile()

    retrieval_cfg = _build(
        RetrievalConfig,
        {
            k: v
            for k, v in retrieval.items()
            if k not in ("bm25", "vector", "reranker")
        }
        | {"bm25": bm25, "vector": vector, "reranker": reranker},
        "retrieval",
    )

    llm_raw["base_url"] = env.get("LLM_API_BASE") or llm_raw.get("base_url") or ""
    llm_raw["api_key"] = env.get("LLM_API_KEY") or None
    llm_raw["model"] = env.get("LLM_MODEL") or llm_raw.get("model") or None
    llm = _build(LLMConfig, llm_raw, "llm")

    eval_section = _build(
        EvalConfig,
        eval_raw | {"dataset": resolve(eval_raw.get("dataset", "data/eval.jsonl"))},
        "eval",
    )
    eval_cfg = EvalConfig(
        dataset=eval_section.dataset,
        k_values=tuple(eval_section.k_values),
        modes=tuple(eval_section.modes),
        embedding_models=tuple(eval_section.embedding_models),
        generation=eval_section.generation,
        results_dir=resolve(eval_raw.get("results_dir", "results")),
        call_log=resolve(eval_raw.get("call_log", "results/calls.jsonl")),
    )
    _need(bool(eval_cfg.k_values), "eval.k_values не може бути порожнім")
    _need(all(k >= 1 for k in eval_cfg.k_values), "eval.k_values має містити значення >= 1")
    _need(
        all(m in RETRIEVAL_MODES for m in eval_cfg.modes),
        f"eval.modes має бути підмножиною {RETRIEVAL_MODES}",
    )
    for name in eval_cfg.embedding_models:
        vector.profile(name)

    return Config(
        corpus_dir=resolve(corpus.get("dir", "data/corpus")),
        corpus_glob=tuple(corpus.get("glob", ["*.md", "*.txt"])),
        cache_dir=resolve(index.get("cache_dir", ".cache")),
        chunking=chunking,
        retrieval=retrieval_cfg,
        llm=llm,
        eval=eval_cfg,
        raw=data,
    )
