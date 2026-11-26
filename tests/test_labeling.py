import pytest

from rag_uk.labeling import LabelError, make_record, next_id

KNOWN = {"d:0000", "d:0001"}


def test_record_has_exactly_required_fields():
    record = make_record("Який строк оплати?", ["d:0000"], KNOWN, set(), "q-0001")
    assert record == {
        "id": "q-0001",
        "question": "Який строк оплати?",
        "gold_chunk_ids": ["d:0000"],
    }


def test_strips_and_deduplicates():
    record = make_record("  Питання?  ", [" d:0000 ", "d:0000", ""], KNOWN, set(), "q-0001")
    assert record["question"] == "Питання?"
    assert record["gold_chunk_ids"] == ["d:0000"]


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question(question):
    with pytest.raises(LabelError):
        make_record(question, ["d:0000"], KNOWN, set(), "q-0001")


def test_duplicate_question_is_case_insensitive():
    with pytest.raises(LabelError):
        make_record("Питання?", ["d:0000"], KNOWN, {"питання?"}, "q-0001")


def test_empty_gold():
    with pytest.raises(LabelError):
        make_record("Питання?", [], KNOWN, set(), "q-0001")


def test_unknown_gold_id():
    with pytest.raises(LabelError):
        make_record("Питання?", ["d:9999"], KNOWN, set(), "q-0001")


@pytest.mark.parametrize(
    "existing,expected",
    [(set(), "q-0001"), ({"q-0001", "q-0003"}, "q-0004"), ({"other"}, "q-0001")],
)
def test_next_id(existing, expected):
    assert next_id(existing) == expected
