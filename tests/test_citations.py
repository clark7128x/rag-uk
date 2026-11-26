import pytest

from rag_uk.citations import citation_report, extract_citations


def test_extracts_ids_in_order():
    text = "Перше «цитата» [d:0001]. Друге «цитата» [d:0002]."
    assert extract_citations(text) == ["d:0001", "d:0002"]


def test_ignores_non_id_brackets():
    assert extract_citations("[джерело] [d:01] [d:00001] []") == []


def test_unicode_doc_id():
    assert extract_citations("«цитата» [договір_2024:0007]") == ["договір_2024:0007"]


def test_deduplicates_repeated_id():
    assert extract_citations("[d:0001] і знову [d:0001]") == ["d:0001"]


def test_no_citations():
    assert extract_citations("Відповідь без посилань.") == []


def test_invalid_input_type():
    with pytest.raises(TypeError):
        extract_citations(None)


def test_report_all_in_context():
    report = citation_report("«ц» [d:0001]", ["d:0001", "d:0002"])
    assert report["all_in_context"] is True
    assert report["unknown_citations"] == []


def test_report_with_unknown_id():
    report = citation_report("«ц» [d:0001] і [d:9999]", ["d:0001"])
    assert report["all_in_context"] is False
    assert report["unknown_citations"] == ["d:9999"]


def test_report_without_citations_is_not_grounded():
    report = citation_report("Недостатньо даних у корпусі.", ["d:0001"])
    assert report["has_citations"] is False
    assert report["all_in_context"] is False
