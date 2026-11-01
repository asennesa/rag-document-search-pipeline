"""Token-aware chunking by character offsets into cleaned text.

Chunk text is always a substring of the source, never decoded from tokens, because the
tokenizer lowercases and strips accents. Each candidate is retokenized before acceptance:
a substring that starts inside a word can tokenize differently from the full text.
"""

from dataclasses import dataclass

from rag_pipeline.errors import ChunkingError
from rag_pipeline.identifiers import make_chunk_id, sha256_text
from rag_pipeline.models import Chunk, Document
from rag_pipeline.tokenization import TextTokenizer

_NO_BOUNDARY, _WORD, _LINE_OR_SENTENCE, _PARAGRAPH = range(4)


@dataclass(frozen=True)
class TextSpan:
    start_char: int
    end_char: int
    text: str
    token_count: int  # Complete encoder input, including special tokens.


class Chunker:
    def __init__(
        self,
        tokenizer: TextTokenizer,
        *,
        chunk_size: int,
        chunk_overlap: int,
        max_input_tokens: int,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be at least 0 and below chunk_size")
        if chunk_size + tokenizer.special_token_count > max_input_tokens:
            raise ValueError("chunk_size plus special tokens exceeds the model input limit")
        self._tokenizer = tokenizer
        self._chunk_size = chunk_size
        self._overlap = chunk_overlap
        self._max_input_tokens = max_input_tokens
        # Overlap may grow slightly so the next chunk starts on a whole word.
        self._max_overlap_extension = min(8, chunk_overlap // 4)
        # Stopping early at a boundary must still let the next chunk move past this one.
        self._min_advance = max(chunk_size // 2, chunk_overlap + self._max_overlap_extension + 1)

    def split_text(self, text: str) -> list[TextSpan]:
        if not text.strip():
            return []
        offsets = self._tokenizer.offsets(text)
        if not offsets:
            raise ChunkingError("Text produced no tokens and cannot be embedded")

        spans: list[TextSpan] = []
        start = 0
        while True:
            end = min(start + self._chunk_size, len(offsets))
            if end < len(offsets):
                end = self._best_end(text, offsets, start, end)
            span, end = self._fit(text, offsets, start, end)
            spans.append(span)
            if end >= len(offsets):
                return spans
            start = self._next_start(offsets, start, end)

    def split_document(self, document: Document, revision_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in document.pages:
            for span in self.split_text(page.cleaned_text):
                content_hash = sha256_text(span.text)
                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(
                            document.document_id,
                            revision_id,
                            page.page_number,
                            len(chunks),
                            content_hash,
                        ),
                        document_id=document.document_id,
                        source=document.source,
                        page_number=page.page_number,
                        chunk_index=len(chunks),
                        text=span.text,
                        content_hash=content_hash,
                        document_hash=document.document_hash,
                        revision_id=revision_id,
                        start_char=span.start_char,
                        end_char=span.end_char,
                        token_count=span.token_count,
                    )
                )
        return chunks

    def _best_end(self, text: str, offsets: list[tuple[int, int]], start: int, limit: int) -> int:
        """Pick the latest cut with the strongest boundary; `limit` < len(offsets)."""
        floor = min(start + self._min_advance, limit)
        best_end, best_rank = limit, _NO_BOUNDARY
        for end in range(limit, floor - 1, -1):
            rank = _boundary_rank(text, offsets[end - 1][1], offsets[end][0])
            if rank > best_rank:
                best_end, best_rank = end, rank
                if rank == _PARAGRAPH:
                    break
        return best_end

    def _fit(
        self, text: str, offsets: list[tuple[int, int]], start: int, end: int
    ) -> tuple[TextSpan, int]:
        """Shrink until the actual substring fits; the token estimate alone is not proof."""
        while True:
            span_text, start_char, end_char = _slice(text, offsets, start, end)
            content_tokens = len(self._tokenizer.offsets(span_text))
            if 0 < content_tokens <= self._chunk_size:
                break
            if end - start <= 1:
                raise ChunkingError("A single token does not fit within the chunk size")
            end = max(start + 1, end - max(content_tokens - self._chunk_size, 1))
        token_count = content_tokens + self._tokenizer.special_token_count
        if token_count > self._max_input_tokens:
            raise ChunkingError("Chunk exceeds the model input token limit")
        return TextSpan(start_char, end_char, span_text, token_count), end

    def _next_start(self, offsets: list[tuple[int, int]], start: int, end: int) -> int:
        candidate = max(end - self._overlap, start + 1)
        lowest = max(start + 1, candidate - self._max_overlap_extension)
        for index in range(candidate, lowest - 1, -1):
            if offsets[index - 1][1] < offsets[index][0]:
                return index
        return candidate


def _boundary_rank(text: str, left_end: int, right_start: int) -> int:
    gap = text[left_end:right_start]
    if "\n\n" in gap:
        return _PARAGRAPH
    if "\n" in gap or (gap and text[left_end - 1] in ".!?"):
        return _LINE_OR_SENTENCE
    return _WORD if gap else _NO_BOUNDARY


def _slice(text: str, offsets: list[tuple[int, int]], start: int, end: int) -> tuple[str, int, int]:
    """Cover characters that produce no tokens, so no non-whitespace text falls in a gap."""
    start_char = offsets[start][0] if start else len(text) - len(text.lstrip())
    if end < len(offsets):
        limit = max(offsets[end - 1][1], offsets[end][0])
    else:
        limit = len(text)
    end_char = len(text[:limit].rstrip())
    return text[start_char:end_char], start_char, end_char
