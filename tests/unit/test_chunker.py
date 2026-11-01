import pytest

from rag_pipeline.errors import ChunkingError
from rag_pipeline.identifiers import sha256_text
from rag_pipeline.ingestion.chunker import Chunker
from rag_pipeline.models import Document, Page


def make_chunker(tokenizer, size=20, overlap=4, limit=512):
    return Chunker(tokenizer, chunk_size=size, chunk_overlap=overlap, max_input_tokens=limit)


def words(count, start=0):
    return " ".join(f"w{i}" for i in range(start, start + count))


def assert_covers_all_text(text, spans):
    covered = [False] * len(text)
    for span in spans:
        assert text[span.start_char : span.end_char] == span.text
        covered[span.start_char : span.end_char] = [True] * len(span.text)
    missing = [i for i, char in enumerate(text) if not char.isspace() and not covered[i]]
    assert missing == []


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (-1, 0), (10, 10), (10, 11), (10, -1), (511, 0)],
)
def test_invalid_configuration_is_rejected(fake_tokenizer, size, overlap):
    with pytest.raises(ValueError):
        make_chunker(fake_tokenizer, size, overlap)


def test_largest_chunk_fits_model_limit_with_special_tokens(fake_tokenizer):
    chunker = make_chunker(fake_tokenizer, 510, 0)
    [*spans] = chunker.split_text(words(1000))
    assert max(span.token_count for span in spans) == 512


@pytest.mark.parametrize("text", ["", "   \n "])
def test_blank_text_gives_no_chunks(fake_tokenizer, text):
    assert make_chunker(fake_tokenizer).split_text(text) == []


def test_short_text_is_one_chunk(fake_tokenizer):
    [span] = make_chunker(fake_tokenizer).split_text("Short text.")
    assert (span.text, span.start_char, span.token_count) == ("Short text.", 0, 5)


def test_exact_size_text_is_one_chunk(fake_tokenizer):
    assert len(make_chunker(fake_tokenizer).split_text(words(20))) == 1


def test_one_token_over_size_gives_two_chunks(fake_tokenizer):
    assert len(make_chunker(fake_tokenizer).split_text(words(21))) == 2


def test_unbroken_words_overlap_by_the_configured_tokens(fake_tokenizer):
    tokens = words(100).split()
    spans = make_chunker(fake_tokenizer, 20, 4).split_text(" ".join(tokens))
    for left, right in zip(spans, spans[1:], strict=False):
        shared = set(left.text.split()) & set(right.text.split())
        assert len(shared) >= 4


def test_zero_overlap_shares_no_tokens_and_loses_none(fake_tokenizer):
    text = words(100)
    spans = make_chunker(fake_tokenizer, 20, 0).split_text(text)
    assert [word for span in spans for word in span.text.split()] == text.split()


@pytest.mark.parametrize(
    "text",
    [
        words(500),
        "\n\n".join(words(30, i * 30) for i in range(10)),
        "Sentence one here. " * 200,
        "x" * 5000,
        "日本語のテキスト。" * 300,
        "é😀 " * 300,
        "[SEP] [CLS] <s> " * 100,
    ],
)
@pytest.mark.parametrize(("size", "overlap"), [(20, 4), (20, 0), (20, 19), (2, 1), (1, 0)])
def test_progress_coverage_and_bounds(fake_tokenizer, text, size, overlap):
    spans = make_chunker(fake_tokenizer, size, overlap).split_text(text)
    assert_covers_all_text(text, spans)
    assert [s.start_char for s in spans] == sorted({s.start_char for s in spans})
    assert all(s.text and s.text == s.text.strip() for s in spans)
    assert all(
        len(fake_tokenizer.offsets(s.text)) <= size
        and s.token_count == len(fake_tokenizer.offsets(s.text)) + 2
        for s in spans
    )


def test_prefers_paragraph_over_sentence_or_word_boundaries(fake_tokenizer):
    text = f"{words(14)}\n\n{words(3)}. {words(20)}"
    first = make_chunker(fake_tokenizer, 20, 2).split_text(text)[0]
    assert first.text == words(14)


def test_prefers_sentence_over_word_boundary(fake_tokenizer):
    text = f"{words(15)}. {words(30)}"
    first = make_chunker(fake_tokenizer, 20, 2).split_text(text)[0]
    assert first.text == f"{words(15)}."


def test_preserves_original_capitalization_and_accents(fake_tokenizer):
    text = "Ünï Çödé KEEPS Its Case. " * 30
    spans = make_chunker(fake_tokenizer).split_text(text.strip())
    assert all(span.text in text for span in spans)
    assert "Ünï" in spans[0].text


class ShiftingTokenizer:
    """Tokens merge unevenly: a substring costs more than its slice of the full text."""

    special_token_count = 2

    def offsets(self, text):
        spans = []
        for start in range(0, len(text), 2):
            spans.append((start, min(start + 2, len(text))))
        if len(text) < 100:  # Candidates retokenize larger, as mid-word cuts can.
            spans = [(i, i + 1) for i in range(len(text))]
        return spans


def test_candidate_is_retokenized_and_shrunk_until_it_fits():
    text = "a" * 400
    tokenizer = ShiftingTokenizer()
    spans = Chunker(tokenizer, chunk_size=20, chunk_overlap=2, max_input_tokens=512).split_text(
        text
    )
    assert all(len(tokenizer.offsets(span.text)) <= 20 for span in spans)
    assert_covers_all_text(text, spans)


class SilentCharacterTokenizer:
    """Like BERT, drops characters such as U+200B without producing a token."""

    special_token_count = 2

    def offsets(self, text):
        return [(i, i + 1) for i, char in enumerate(text) if char not in " \u200b"]


def test_characters_without_tokens_are_not_lost_between_chunks():
    text = "\u200b" + "ab\u200b" * 50 + "\u200b"
    tokenizer = SilentCharacterTokenizer()
    for overlap in (0, 3):
        spans = Chunker(
            tokenizer, chunk_size=10, chunk_overlap=overlap, max_input_tokens=512
        ).split_text(text)
        assert_covers_all_text(text, spans)


def test_text_without_any_tokens_is_an_error_not_silent_loss():
    with pytest.raises(ChunkingError):
        Chunker(
            SilentCharacterTokenizer(), chunk_size=10, chunk_overlap=0, max_input_tokens=512
        ).split_text("\u200b\u200b")


def make_document(*pages):
    return Document(
        document_id="doc",
        source_namespace="ns",
        source="a.pdf",
        filename="a.pdf",
        file_type="pdf",
        document_hash="hash",
        size_bytes=1,
        modified_at="2026-01-01T00:00:00Z",
        extractor="test",
        pages=tuple(
            Page(page_number=number, raw_text=text, cleaned_text=text) for number, text in pages
        ),
    )


def test_document_chunks_keep_page_metadata_and_never_cross_pages(fake_tokenizer):
    document = make_document((1, words(30)), (2, ""), (3, words(5, 100)))
    chunks = make_chunker(fake_tokenizer).split_document(document, "rev")
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert {chunk.page_number for chunk in chunks} == {1, 3}
    last = chunks[-1]
    assert last.text == words(5, 100) and last.page_number == 3
    assert last.content_hash == sha256_text(last.text)
    assert (last.document_id, last.revision_id, last.source) == ("doc", "rev", "a.pdf")


def test_repeated_chunking_gives_identical_ids(fake_tokenizer):
    document = make_document((1, words(100)))
    chunker = make_chunker(fake_tokenizer)
    first = chunker.split_document(document, "rev")
    assert first == chunker.split_document(document, "rev")
    assert len({chunk.chunk_id for chunk in first}) == len(first)
    assert {c.chunk_id for c in first}.isdisjoint(
        c.chunk_id for c in chunker.split_document(document, "rev2")
    )
