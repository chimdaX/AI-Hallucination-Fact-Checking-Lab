import pytest

from hallucination_lab.verification.rationale import check_rationale, rationale_errors

CLAIM = "The Eiffel Tower was completed in 1899."
PASSAGE = ["whose company designed and built the tower from 1887 to 1889"]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "n/a",
        "N/A",
        "tbd",
        "TODO",
        "-",
        "...",
        "?",
        "idk",
        "lorem ipsum",
        "4",
        "4/5",
        "80%",
        "Score: 8",
        "8.5 out of 10",
        "3 stars",
        "!!!",
        "aaaa",
        "Correct.",
        "wrong",
        "TRUE",
        "ok",
    ],
)
def test_invalid_rationales_rejected(text):
    assert rationale_errors(text)
    assert not check_rationale(text, "SUPPORTED", CLAIM, PASSAGE).ok


@pytest.mark.parametrize(
    "text",
    ["The claim is correct.", "Looks fine.", "The AI is wrong.", "Good evidence.", "Verified."],
)
def test_generic_rationales_warned_not_rejected(text):
    result = check_rationale(text, "CONTRADICTED", CLAIM, PASSAGE)
    assert result.ok
    assert any("generic" in w for w in result.warnings)


def test_short_specific_rationale_is_valid_without_warnings():
    result = check_rationale("Source dates it 1887-1889, not 1899.", "CONTRADICTED", CLAIM, PASSAGE)
    assert result.ok and result.warnings == []


def test_rationale_unrelated_to_claim_and_evidence_warned():
    result = check_rationale("Bananas are yellow fruits.", "SUPPORTED", CLAIM, PASSAGE)
    assert any("does not mention" in w for w in result.warnings)


def test_insufficient_evidence_should_say_what_is_missing():
    assert check_rationale("Paris is nice.", "INSUFFICIENT_EVIDENCE").warnings
    assert not check_rationale(
        "No retrieved passage mentions this group or any 2040 publication.",
        "INSUFFICIENT_EVIDENCE",
    ).warnings


def test_not_checkable_is_not_for_hard_research():
    result = check_rationale("I could not find sources for this.", "NOT_CHECKABLE")
    assert any("INSUFFICIENT_EVIDENCE" in w for w in result.warnings)
    assert not check_rationale("A subjective aesthetic opinion.", "NOT_CHECKABLE").warnings
    assert check_rationale("Paris is large.", "NOT_CHECKABLE").warnings
