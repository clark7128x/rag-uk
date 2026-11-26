import pytest

from rag_uk.jsonio import append_jsonl, read_jsonl, write_jsonl


def test_round_trip_with_unicode(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"id": "q-0001", "question": "Питання?"}])
    append_jsonl(path, {"id": "q-0002", "question": "Ще?"})
    rows = read_jsonl(path)
    assert [r["id"] for r in rows] == ["q-0001", "q-0002"]
    assert rows[0]["question"] == "Питання?"


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('\n{"id": "a"}\n\n', encoding="utf-8")
    assert read_jsonl(path) == [{"id": "a"}]


def test_empty_file(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("", encoding="utf-8")
    assert read_jsonl(path) == []


@pytest.mark.parametrize("content", ["{не json}\n", "[1, 2]\n"])
def test_malformed_lines(tmp_path, content):
    path = tmp_path / "rows.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        read_jsonl(path)
