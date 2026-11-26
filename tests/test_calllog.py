import json

from rag_uk.calllog import CallLog


def test_summary_aggregates_tokens_latency_and_cost(tmp_path):
    log = CallLog(tmp_path / "calls.jsonl", price_per_1m_input=2.0, price_per_1m_output=4.0)
    log.record(kind="chat", model="m", input_tokens=1000, output_tokens=500, latency_s=1.0)
    log.record(kind="chat", model="m", input_tokens=1000, output_tokens=500, latency_s=3.0)
    summary = log.summary()
    row = summary["by_model"][0]
    assert row["calls"] == 2
    assert row["input_tokens"] == 2000
    assert row["output_tokens"] == 1000
    assert row["latency_p50_s"] == 1.0
    assert row["latency_p95_s"] == 3.0
    assert row["cost"] == (2000 / 1e6 * 2.0) + (1000 / 1e6 * 4.0)
    assert summary["cost_total"] == row["cost"]


def test_cost_is_none_without_prices():
    log = CallLog()
    log.record(kind="chat", model="m", input_tokens=10, output_tokens=10, latency_s=0.1)
    assert log.summary()["by_model"][0]["cost"] is None


def test_failed_call_without_tokens():
    log = CallLog(price_per_1m_input=1.0, price_per_1m_output=1.0)
    log.record(
        kind="chat", model="m", input_tokens=None, output_tokens=None,
        latency_s=0.5, ok=False, error="http 429",
    )
    summary = log.summary()
    assert summary["failed"] == 1
    assert summary["by_model"][0]["input_tokens"] is None
    assert summary["by_model"][0]["cost"] is None


def test_embedding_and_rerank_have_no_cost():
    log = CallLog(price_per_1m_input=1.0, price_per_1m_output=1.0)
    log.record(kind="embedding", model="e", input_tokens=100, output_tokens=0, latency_s=0.2)
    log.record(kind="rerank", model="r", input_tokens=100, output_tokens=0, latency_s=0.3)
    rows = {r["kind"]: r for r in log.summary()["by_model"]}
    assert rows["embedding"]["cost"] is None
    assert rows["rerank"]["output_tokens"] == 0
    assert log.summary()["cost_total"] is None


def test_empty_log():
    summary = CallLog().summary()
    assert summary["calls"] == 0 and summary["by_model"] == []


def test_records_are_appended_to_jsonl(tmp_path):
    path = tmp_path / "nested" / "calls.jsonl"
    log = CallLog(path)
    log.record(kind="chat", model="m", input_tokens=1, output_tokens=2, latency_s=0.1)
    log.record(kind="chat", model="m", input_tokens=3, output_tokens=4, latency_s=0.2)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[0]["model"] == "m" and lines[1]["input_tokens"] == 3
