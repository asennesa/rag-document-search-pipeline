import pytest

from rag_pipeline.ingestion.text_cleaner import clean_text


def test_normalizes_line_endings_blank_runs_and_trailing_space():
    assert clean_text("\ufeffTitle  \r\n\r\n\r\n\r\nBody\rNext\x0cPage\n") == (
        "Title\n\nBody\nNext\nPage"
    )


def test_preserves_meaning_and_layout():
    text = "# Heading\n\nDo not share 25.5% of data.\n\n    if x:\n        y  =  1\n\nnaïve 日本語"
    assert clean_text(text) == text


def test_removes_control_characters_but_keeps_tabs():
    assert clean_text("a\x00b\tc\x7f") == "ab\tc"


@pytest.mark.parametrize("raw", ["", "  \n\t\r\n ", "\ufeff"])
def test_whitespace_only_becomes_empty(raw):
    assert clean_text(raw) == ""


@pytest.mark.parametrize("raw", ["a \r\n\r\n\r\nb\x0c", "\ufeff x\x00\n\n\n y  ", "a\x0b \nb"])
def test_cleaning_is_idempotent(raw):
    once = clean_text(raw)
    assert clean_text(once) == once
