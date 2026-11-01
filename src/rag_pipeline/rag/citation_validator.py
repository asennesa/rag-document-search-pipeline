"""Check that cited IDs refer to sources actually sent to the model.

A valid citation proves the ID exists. It does not prove the source supports the claim,
so a passing result must never be presented as factual correctness.
"""

import re
from collections.abc import Collection, Sequence

from rag_pipeline.models import CitationResult

_LISTED_ID = re.compile(r"S\d+")
_VALID_GROUP = re.compile(r"\[\s*S\d+(?:\s*,\s*S\d+)*\s*\]")
_SOURCE_ID = re.compile(r"S\d+")
_LOOKS_LIKE_CITATION = re.compile(r"\[\s*(?:S\s*\d|S\s*(?=\])|sources?\b)[^\]]*\]", re.IGNORECASE)


def validate_citations(
    answer: str,
    included_ids: Collection[str],
    *,
    insufficient_evidence: bool,
    listed_ids: Sequence[str] = (),
) -> CitationResult:
    """Check inline citations and the model's listed IDs alike.

    Unknown IDs are reported and never repaired by guessing.
    """
    cited: list[str] = [item for item in dict.fromkeys(listed_ids) if _LISTED_ID.fullmatch(item)]
    for group in _VALID_GROUP.findall(answer):
        for source_id in _SOURCE_ID.findall(group):
            if source_id not in cited:
                cited.append(source_id)
    malformed = tuple(
        match.group()
        for match in _LOOKS_LIKE_CITATION.finditer(answer)
        if not _VALID_GROUP.fullmatch(match.group())
    ) + tuple(item for item in listed_ids if not _LISTED_ID.fullmatch(item))
    unknown = tuple(source_id for source_id in cited if source_id not in included_ids)

    if unknown or malformed:
        status = "invalid"
    elif cited:
        status = "valid"
    elif insufficient_evidence:
        status = "not_applicable"  # A refusal makes no claim that needs a source.
    else:
        status = "missing"  # No citations must never count as vacuously valid.
    return CitationResult(
        status=status, cited_ids=tuple(cited), unknown_ids=unknown, malformed=malformed
    )
