from rag_uk.report import MISSING, render_markdown


def payload(metrics=None, generation=None):
    return {
        "config": {"eval": {"k_values": [1, 3, 5]}},
        "dataset": {"path": "data/eval.jsonl", "rows": 2, "skipped": 0},
        "retrieval": [
            {"mode": "bm25", "embedding_model": None, "metrics": metrics, "latency_s": None}
        ],
        "generation": generation or {},
        "calls": {"by_model": []},
    }


def test_unmeasured_cells():
    markdown = render_markdown(payload())
    assert markdown.count(MISSING) >= 5
    assert "| Режим | Ембединги | recall@1 | recall@3 | recall@5 | MRR |" in markdown


def test_measured_cells_are_formatted():
    metrics = {"recall@1": 0.5, "recall@3": 0.75, "recall@5": 1.0, "mrr": 0.625}
    markdown = render_markdown(payload(metrics))
    assert "0.500" in markdown and "0.625" in markdown


def test_generation_skip_reason():
    markdown = render_markdown(payload(generation={"skipped_reason": "не задано LLM_MODEL"}))
    assert "Пропущено: не задано LLM_MODEL." in markdown


def test_variant_error_is_reported():
    data = payload()
    data["retrieval"][0]["error"] = "RuntimeError: немає sentence-transformers"
    assert "немає sentence-transformers" in render_markdown(data)


def test_empty_retrieval_list():
    data = payload()
    data["retrieval"] = []
    assert "## Пошук" in render_markdown(data)
