import pytest

from rag_uk.config import ConfigError, load_config

MINIMAL = """
corpus:
  dir: corpus
index:
  cache_dir: cache
retrieval:
  mode: bm25
  top_k: 3
  candidate_k: 10
"""


def write(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_minimal_and_resolves_paths(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL), env={})
    assert cfg.retrieval.mode == "bm25"
    assert cfg.corpus_dir == (tmp_path / "corpus").resolve()
    assert cfg.cache_dir == (tmp_path / "cache").resolve()
    assert cfg.eval.dataset == (tmp_path / "data/eval.jsonl").resolve()


def test_env_overrides(tmp_path):
    path = write(tmp_path, MINIMAL)
    cfg = load_config(
        path,
        env={
            "RAG_RETRIEVAL_MODE": "hybrid",
            "RAG_EMBEDDING_MODEL": "minilm-l12",
            "LLM_API_BASE": "http://localhost:8000/v1",
            "LLM_API_KEY": "secret",
            "LLM_MODEL": "local-model",
        },
    )
    assert cfg.retrieval.mode == "hybrid"
    assert cfg.retrieval.vector.model == "minilm-l12"
    assert cfg.llm.configured and cfg.llm.api_key == "secret"


def test_llm_not_configured_by_default(tmp_path):
    assert load_config(write(tmp_path, MINIMAL), env={}).llm.configured is False


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "absent.yaml", env={})


@pytest.mark.parametrize(
    "text",
    [
        "retrieval:\n  mode: magic\n",
        "retrieval:\n  top_k: 5\n  candidate_k: 2\n",
        "chunking:\n  overlap_char: 10\n",
        "chunking:\n  target_chars: 100\n  overlap_chars: 100\n",
        "retrieval:\n  vector:\n    model: nope\n",
        "eval:\n  k_values: [0]\n",
        "eval:\n  modes: [magic]\n",
        "eval:\n  embedding_models: [nope]\n",
        "llm:\n  price_per_1m_input: -1\n",
    ],
)
def test_invalid_configs(tmp_path, text):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, text), env={})


def test_unknown_embedding_profile_lookup(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL), env={})
    with pytest.raises(ConfigError):
        cfg.retrieval.vector.profile("nope")
