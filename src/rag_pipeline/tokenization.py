"""Token counting that matches the embedding model and can never truncate.

FastEmbed enables truncation on its own tokenizer, so counting with that instance would
report at most 512 tokens for any input. This separate instance disables truncation and
padding, which is what makes the "never silently discard content" guarantee checkable.
"""

from pathlib import Path
from typing import Protocol

from tokenizers import Tokenizer

# BGE v1.5 retrieval instruction. Questions get it; documents never do. FastEmbed's
# query_embed adds nothing for this model, so the query path must add it exactly once.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class TextTokenizer(Protocol):
    """The tokenizer behavior the chunker depends on."""

    @property
    def special_token_count(self) -> int:
        """Tokens the encoder adds around content, such as [CLS] and [SEP]."""
        ...

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Character span in `text` of every content token, without special tokens."""
        ...


class EmbeddingTokenizer:
    def __init__(self, tokenizer_file: Path) -> None:
        self._tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()

    @property
    def special_token_count(self) -> int:
        return self._tokenizer.num_special_tokens_to_add(is_pair=False)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return list(self._tokenizer.encode(text, add_special_tokens=False).offsets)
