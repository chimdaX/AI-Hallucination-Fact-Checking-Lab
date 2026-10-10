import pytest

from hallucination_lab.errors import ExtractionError
from hallucination_lab.extraction import ExtractionConfig, RuleBasedClaimExtractor, extract_claims
from hallucination_lab.extraction.features import detect_features, temporal_context
from hallucination_lab.extraction.rules import classify, normalize_claim_text
from hallucination_lab.models import AIResponse, ExtractionMethod, NonCheckableReason


def _claims(text: str, **cfg):
    return extract_claims(text, ExtractionConfig(**cfg) if cfg else None)[1]


def test_original_wording_and_spans_preserved():
    text = "Additionally,  the Moon orbits Earth. Mars has two moons, and Venus has none."
    response, claims = extract_claims(text)
    for claim in claims:
        assert response.text[claim.span.start : claim.span.end] == claim.original_text
        assert claim.extraction_method is ExtractionMethod.RULE_BASED
        assert claim.extraction_confidence is None  # rules are not calibrated
    assert claims[0].original_text == "Additionally,  the Moon orbits Earth."
    assert claims[0].normalized_claim == "The Moon orbits Earth."


def test_multiple_assertions_are_separated():
    claims = _claims("Mars has two moons, and Venus has no moons. Jupiter is a gas giant.")
    assert [c.normalized_claim for c in claims] == [
        "Mars has two moons",
        "Venus has no moons.",
        "Jupiter is a gas giant.",
    ]
    assert "compound sentence split" in claims[0].extraction_notes[0]


def test_compound_splitting_can_be_disabled():
    claims = _claims("Mars has two moons, and Venus has no moons.", split_compound_sentences=False)
    assert len(claims) == 1


def test_extraction_is_deterministic():
    text = "The Eiffel Tower is in Paris. It opened in 1889."
    assert _claims(text) == _claims(text)


@pytest.mark.parametrize(
    "text,reason",
    [
        ("Is the Eiffel Tower in Paris?", NonCheckableReason.QUESTION),
        ("I think Paris is the capital of France.", NonCheckableReason.OPINION),
        ("Paris is the most beautiful city.", NonCheckableReason.OPINION),
        ("Everyone should visit Rome at least once.", NonCheckableReason.OPINION),
        ("Make sure to check the official website.", NonCheckableReason.INSTRUCTION),
        ("You should consult a doctor about this.", NonCheckableReason.INSTRUCTION),
        ("If the Moon disappeared, tides would be much weaker.", NonCheckableReason.HYPOTHETICAL),
        ("Imagine a city with ten million bicycles.", NonCheckableReason.HYPOTHETICAL),
        ("Great question!", NonCheckableReason.NO_ASSERTION),
        ("Here are the key facts:", NonCheckableReason.NO_ASSERTION),
        ("Absolutely.", NonCheckableReason.NO_ASSERTION),
    ],
)
def test_non_factual_segments_are_not_treated_as_facts(text, reason):
    (claim,) = _claims(text)
    assert claim.requires_verification is False
    assert claim.non_checkable_reason is reason
    assert reason.value in claim.extraction_notes[-1]  # the firing rule is recorded


@pytest.mark.parametrize(
    "text",
    [
        "Einstein is best known for the theory of relativity.",
        "Use of coal declined in the 2010s.",
        "Note that the Pacific is the largest ocean.",
        "Water boils at 100 °C at sea level.",
    ],
)
def test_factual_sentences_not_falsely_flagged(text):
    (claim,) = _claims(text)
    assert claim.requires_verification is True
    assert claim.non_checkable_reason is None


def test_numbers_dates_measurements_entities():
    f = detect_features(
        "On 14 July 1789, crowds in Paris stormed the Bastille; the city had about "
        "600,000 people and the river is 777 km long, a 12.5% share."
    )
    assert "14 July 1789" in f.dates
    assert "600,000" in f.numbers and "12.5%" in f.numbers
    assert "777 km" in f.measurements and "600,000 people" in f.measurements
    assert {"Paris", "Bastille"} <= set(f.named_entities)


def test_measured_number_is_not_mistaken_for_year():
    f = detect_features("The summit is 1500 metres high.")
    assert f.dates == []
    assert f.measurements == ["1500 metres"]


def test_citation_and_quote_detection():
    f = detect_features(
        "A study by Hartwell et al. (2019) in doi:10.1234/abc.567 found it. "
        'Einstein said "imagination is more important than knowledge".'
    )
    assert any("Hartwell et al" in c for c in f.citations)
    assert any(c.startswith("doi:10.1234") for c in f.citations)
    assert f.quotations == ["imagination is more important than knowledge"]


def test_temporal_context_recorded():
    (claim,) = _claims("Currently, the tallest building is the Burj Khalifa.")
    assert claim.features.temporal_markers == ["currently"]
    assert "time-sensitive" in claim.temporal_context


def test_temporal_context_none_when_no_time_reference():
    assert temporal_context(detect_features("Gold is a metal.")) is None


def test_pronoun_start_is_flagged_for_context():
    claims = _claims("The Eiffel Tower is in Paris. It opened in 1889.")
    assert claims[1].features.starts_with_pronoun
    assert any("context" in n for n in claims[1].extraction_notes)


def test_normalization_does_not_change_meaning():
    assert normalize_claim_text("  Furthermore,   it is   large. ") == "It is large."
    assert normalize_claim_text("However,") == "However,"  # never produces empty text


def test_classify_returns_describable_result():
    result = classify("Paris is the capital of France.")
    assert result.is_checkable and "no non-checkable rule" in result.describe()


@pytest.mark.parametrize("text", ["", "   \n"])
def test_empty_input_raises_clear_error(text):
    with pytest.raises(ExtractionError):
        extract_claims(text)


def test_punctuation_only_response_yields_only_non_checkable():
    response = AIResponse.from_text("...")
    claims = RuleBasedClaimExtractor().extract(response)
    assert all(not c.requires_verification for c in claims)


def test_unicode_text_handled():
    response, claims = extract_claims("Zürich liegt in der Schweiz. São Paulo is in Brazil.")
    assert len(claims) == 2
    assert response.text[claims[1].span.start : claims[1].span.end] == "São Paulo is in Brazil."
