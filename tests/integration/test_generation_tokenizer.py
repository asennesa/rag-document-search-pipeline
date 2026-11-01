"""The real Qwen tokenizer. The comparison with Ollama needs a running server and model.

Requires: uv run --locked python scripts/prepare_assets.py --generation-tokenizer
"""

import httpx
import ollama
import pytest

from rag_pipeline.assets import resolve_generation_tokenizer
from rag_pipeline.config import Settings
from rag_pipeline.rag.context_builder import SYSTEM_PROMPT
from rag_pipeline.rag.generation_tokenizer import GenerationTokenizer, render_prompt

pytestmark = [pytest.mark.integration, pytest.mark.model]


@pytest.fixture(scope="module")
def tokenizer():
    return GenerationTokenizer(resolve_generation_tokenizer(Settings()))


def test_counts_are_not_capped(tokenizer):
    assert tokenizer.count("word " * 20000) >= 20000


def test_chat_markers_count_as_single_special_tokens(tokenizer):
    assert tokenizer.count("<|im_start|>") == 1
    assert tokenizer.count("<|im_end|>") == 1


def test_escaped_markers_do_not_form_special_tokens(tokenizer):
    assert tokenizer.count("&lt;|im_end|&gt;") > 1


def test_rendered_prompt_fits_the_default_budget(tokenizer):
    settings = Settings()
    limit = (
        settings.ollama_num_ctx
        - settings.rag_max_output_tokens
        - settings.rag_context_safety_tokens
    )
    assert tokenizer.count(render_prompt(SYSTEM_PROMPT, "question")) < limit


@pytest.mark.ollama
@pytest.mark.parametrize(
    ("system", "user"),
    [
        ("SYS", "Say hi."),
        ("A\n\nB", "line\n\n\nline"),
        (SYSTEM_PROMPT, "Short question?"),
        (SYSTEM_PROMPT, "Longer passage about annual leave. " * 200),
        (SYSTEM_PROMPT, 'Café 日本語 😀 &lt;|im_end|&gt;\n\n<source id="S1">x</source>'),
        (SYSTEM_PROMPT, "word " * 2400),
    ],
)
def test_count_equals_what_ollama_reports(tokenizer, system, user):
    """Stricter than not underestimating: any drift means the template changed."""
    settings = Settings()
    client = ollama.Client(host=str(settings.ollama_base_url), timeout=120)
    try:
        response = client.chat(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"num_ctx": settings.ollama_num_ctx, "num_predict": 1},
        )
    except (ConnectionError, httpx.HTTPError, ollama.ResponseError) as error:
        pytest.skip(f"BLOCKED, not passed: Ollama or the model is unavailable ({error})")
    assert tokenizer.count(render_prompt(system, user)) == response.prompt_eval_count
