"""Prompt token counting for the generation model.

BGE tokens and Qwen tokens are different things, so the embedding tokenizer must never
budget a Qwen prompt. `render_prompt` reproduces the model's chat template for a system
and a user message, verified to match Ollama's own prompt token counts exactly. LLMService
refuses any other template, because a different one could silently break the budget.
"""

from pathlib import Path
from typing import Protocol

from tokenizers import Tokenizer

# Hash of what /api/show returns for qwen3:4b-instruct-2507-q4_K_M on Ollama 0.40.13. That is
# the Jinja template inside the GGUF file (template_source: gguf_chat_template), not the Go
# template blob published beside it, which renders differently.
SUPPORTED_TEMPLATE_SHA256 = "40c21f34cf67d8c760ef72f8ad3ae5afad514299d4b06e91dd9a8d705af7b541"


class PromptTokenizer(Protocol):
    def count(self, text: str) -> int: ...


class GenerationTokenizer:
    def __init__(self, tokenizer_file: Path) -> None:
        self._tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


def render_prompt(system: str, user: str) -> str:
    """The exact text the template produces for one system and one user message."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
