import pytest

from rag_uk.chunking import Document, chunk_document, split_paragraphs, split_sentences
from rag_uk.config import ChunkingConfig, ConfigError

CONTRACT = """# Договір

## 3. Оплата

Оплата здійснюється протягом 10 банківських днів.
Виняток: для державних замовників строк становить 30 днів.
"""


def make_cfg(target: int, overlap: int = 0, hard_max: int | None = None) -> ChunkingConfig:
    return ChunkingConfig(
        target_chars=target,
        overlap_chars=overlap,
        min_chars=0,
        hard_max_chars=hard_max or target * 2,
    )


def test_exception_separated_from_rule_with_small_target():
    chunks, _ = chunk_document(Document("doc", CONTRACT), make_cfg(60, hard_max=120))
    assert len(chunks) == 2
    assert "10 банківських" in chunks[0].text
    assert "Виняток" not in chunks[0].text
    assert "Виняток" in chunks[1].text


def test_exception_kept_with_rule_when_target_large():
    chunks, _ = chunk_document(Document("doc", CONTRACT), make_cfg(900))
    assert len(chunks) == 1
    assert "10 банківських" in chunks[0].text
    assert "Виняток" in chunks[0].text


def test_section_and_paragraph_metadata():
    chunks, paragraphs = chunk_document(Document("doc", CONTRACT), make_cfg(900))
    assert chunks[0].section == "Договір > 3. Оплата"
    assert chunks[0].paragraph_ids() == ["doc:p000"]
    assert paragraphs[0].chunk_ids == ["doc:0000"]


def test_offsets_round_trip():
    text = "# Заголовок\n\nПерший абзац з двох речень. Друге речення.\n\nДругий абзац.\n"
    doc = Document("doc", text)
    chunks, paragraphs = chunk_document(doc, make_cfg(900))
    for item in chunks + paragraphs:
        assert text[item.char_start : item.char_end] == item.text


def test_overlap_repeats_whole_sentence():
    text = "Ааа ааа ааа. Ббб ббб ббб. Ввв ввв ввв. Ггг ггг ггг."
    chunks, _ = chunk_document(Document("d", text), make_cfg(26, overlap=14, hard_max=40))
    assert len(chunks) == 3
    assert chunks[0].text == "Ааа ааа ааа. Ббб ббб ббб."
    assert chunks[1].text.startswith("Ббб ббб ббб.")


def test_paragraph_boundary_starts_new_chunk():
    text = "Ааа ааа.\n\nБбб ббб."
    chunks, _ = chunk_document(Document("d", text), make_cfg(100))
    assert [c.text for c in chunks] == ["Ааа ааа.", "Ббб ббб."]


def test_hard_split_respects_limit_and_word_boundaries():
    text = " ".join(["слово"] * 60)
    chunks, _ = chunk_document(Document("d", text), make_cfg(100, hard_max=100))
    assert len(chunks) >= 3
    assert all(len(c.text) <= 100 for c in chunks)
    assert all(not c.text.startswith(" ") and not c.text.endswith(" ") for c in chunks)


def test_hard_split_of_single_long_word():
    chunks, _ = chunk_document(Document("d", "а" * 250), make_cfg(100, hard_max=100))
    assert [len(c.text) for c in chunks] == [100, 100, 50]


def test_empty_and_heading_only_documents():
    assert chunk_document(Document("d", ""), make_cfg(100)) == ([], [])
    assert chunk_document(Document("d", "# Тільки заголовок\n"), make_cfg(100)) == ([], [])


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Згідно зі ст. 5 договору оплата триває 10 днів. Виняток діє.", 2),
        ("І. Я. Франко написав це. Далі йде нове речення.", 2),
        ("1. Перший пункт\n2. Другий пункт", 1),
        ("Одне речення без крапки", 1),
        ("", 0),
    ],
)
def test_sentence_boundaries(text, expected):
    assert len(split_sentences(text)) == expected


def test_paragraph_indices_are_sequential():
    paragraphs = split_paragraphs("d", "Перший.\n\n# Розділ\n\nДругий.\n\nТретій.")
    assert [p.index for p in paragraphs] == [0, 1, 2]
    assert paragraphs[1].section == "Розділ"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_chars": 0},
        {"target_chars": 100, "overlap_chars": 100},
        {"target_chars": 100, "overlap_chars": -1},
        {"target_chars": 100, "hard_max_chars": 50},
        {"target_chars": 100, "min_chars": 200},
    ],
)
def test_invalid_chunking_config(kwargs):
    with pytest.raises(ConfigError):
        ChunkingConfig(**kwargs)


def test_chunk_does_not_cross_section_boundary():
    text = "# А\n\nКоротке правило.\n\n# Б\n\nІнша тема тут.\n"
    cfg = ChunkingConfig(target_chars=900, overlap_chars=150, min_chars=120, hard_max_chars=1400)
    chunks, _ = chunk_document(Document("d", text), cfg)
    assert [c.section for c in chunks] == ["А", "Б"]
    assert all("#" not in c.text for c in chunks)
    assert chunks[0].text == "Коротке правило."
    assert chunks[1].text == "Інша тема тут."


def test_no_overlap_across_section_boundary():
    text = "# А\n\nАаа ааа. Ббб ббб.\n\n# Б\n\nВвв ввв. Ггг ггг.\n"
    cfg = ChunkingConfig(target_chars=900, overlap_chars=150, min_chars=0, hard_max_chars=1400)
    chunks, _ = chunk_document(Document("d", text), cfg)
    assert [c.text for c in chunks] == ["Ааа ааа. Ббб ббб.", "Ввв ввв. Ггг ггг."]
