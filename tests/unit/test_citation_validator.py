import pytest

from rag_pipeline.rag.citation_validator import validate_citations

INCLUDED = {"S1", "S2"}


def check(answer, refused=False, included=INCLUDED):
    return validate_citations(answer, included, insufficient_evidence=refused)


@pytest.mark.parametrize(
    ("answer", "cited"),
    [
        ("Employees get 25 days [S1].", ("S1",)),
        ("A [S1] and B [S2].", ("S1", "S2")),
        ("Both say so [S1, S2].", ("S1", "S2")),
        ("Repeated [S2] and again [S2].", ("S2",)),
    ],
)
def test_valid_citations(answer, cited):
    result = check(answer)
    assert result.status == "valid"
    assert result.cited_ids == cited


def test_unknown_id_is_invalid_and_never_repaired():
    result = check("Made up [S3].")
    assert result.status == "invalid"
    assert result.unknown_ids == ("S3",)


def test_one_bad_citation_invalidates_good_ones():
    assert check("Good [S1], bad [S9].").status == "invalid"


def test_retrieved_but_excluded_passage_has_no_valid_id():
    """Only passages actually sent get IDs, so citing a dropped one is unknown."""
    assert check("From the omitted one [S2].", included={"S1"}).unknown_ids == ("S2",)


@pytest.mark.parametrize(
    "citation",
    [
        "[Source: annual_leave_policy.md]",
        "[source 1]",
        "[S1-S2]",
        "[S1; S2]",
        "[S]",
        "[s1 and s2]",
        "[S1, annual_leave.md]",
    ],
)
def test_malformed_citations_are_invalid(citation):
    result = check(f"Claim {citation}.")
    assert result.status == "invalid"
    assert result.malformed == (citation,)


def test_substantive_answer_without_citations_is_missing_not_valid():
    assert check("Employees get 25 days.").status == "missing"


def test_refusal_needs_no_citation():
    assert check("The documents do not say.", refused=True).status == "not_applicable"


def test_refusal_with_a_fabricated_citation_still_fails():
    assert check("Nothing found [S7].", refused=True).status == "invalid"


def test_ordinary_brackets_are_not_citations():
    assert check("See clause [3] of the guide [S1].").status == "valid"


def test_bare_words_are_not_citations():
    assert check("The S3 bucket is covered [S1].").status == "valid"


def test_listed_ids_count_as_citations():
    result = validate_citations(
        "25 days.", INCLUDED, insufficient_evidence=False, listed_ids=["S2"]
    )
    assert (result.status, result.cited_ids) == ("valid", ("S2",))


def test_listed_and_inline_ids_merge_without_repeats():
    result = validate_citations(
        "25 days [S1].", INCLUDED, insufficient_evidence=False, listed_ids=["S2", "S1", "S2"]
    )
    assert set(result.cited_ids) == {"S1", "S2"} and len(result.cited_ids) == 2


def test_unknown_listed_id_is_invalid():
    result = validate_citations(
        "25 days.", INCLUDED, insufficient_evidence=False, listed_ids=["S9"]
    )
    assert (result.status, result.unknown_ids) == ("invalid", ("S9",))


@pytest.mark.parametrize("item", ["annual_leave_policy.md", "s1", "S1 ", "[S1]", ""])
def test_malformed_listed_id_is_invalid(item):
    result = validate_citations(
        "25 days.", INCLUDED, insufficient_evidence=False, listed_ids=[item]
    )
    assert result.status == "invalid"
    assert result.malformed == (item,)


def test_empty_list_on_a_real_answer_is_missing():
    assert validate_citations("25 days.", INCLUDED, insufficient_evidence=False).status == "missing"
