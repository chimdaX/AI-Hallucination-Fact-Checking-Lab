import pytest

from hallucination_lab.extraction.segmenter import Segment, split_compound, split_sentences


def texts(text: str) -> list[str]:
    return [s.text for s in split_sentences(text)]


def test_offsets_always_point_at_original_text():
    text = "  First sentence here.  Second one follows!\n\n- A bullet item.\n## Heading\nEnd."
    for seg in split_sentences(text):
        assert text[seg.start : seg.end] == seg.text


def test_basic_split():
    assert texts("Paris is in France. Berlin is in Germany! Is Rome in Italy?") == [
        "Paris is in France.",
        "Berlin is in Germany!",
        "Is Rome in Italy?",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Dr. Smith measured it at approx. 5 metres.",
        "It was proposed by J. R. R. Tolkien in the novel.",
        "Pi is about 3.14159 in value.",
        "Fruits, e.g. apples, contain sugar.",
        "The U.S. economy grew last year.",
        "See Fig. 3 for details.",
    ],
)
def test_abbreviations_and_decimals_do_not_split(text):
    assert texts(text) == [text]


def test_sentence_final_etc_splits_before_capital():
    assert texts("Apples, pears, etc. They are fruit.") == [
        "Apples, pears, etc.",
        "They are fruit.",
    ]


def test_numeric_abbreviation_only_before_number():
    assert texts("The answer is no. Paris is the capital.") == [
        "The answer is no.",
        "Paris is the capital.",
    ]


def test_bullets_are_separate_and_markers_excluded():
    text = "Key facts:\n- Water boils at 100 °C\n- Iron melts at 1538 °C\n1. Gold is a metal"
    assert texts(text) == [
        "Key facts:",
        "Water boils at 100 °C",
        "Iron melts at 1538 °C",
        "Gold is a metal",
    ]


def test_soft_wrapped_lines_are_joined():
    text = "The Nile is often described as\nthe longest river in Africa."
    segs = split_sentences(text)
    assert len(segs) == 1 and segs[0].text == text


def test_closing_quote_stays_with_sentence():
    assert texts('He said "it works." Then he left.') == ['He said "it works."', "Then he left."]


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_empty_input(text):
    assert split_sentences(text) == []


def _compound(text: str) -> list[str]:
    return [s.text for s in split_compound(text, Segment(text, 0, len(text)))]


def test_compound_split_on_independent_clauses():
    text = "The formula of water is H2O, and each molecule contains three atoms."
    assert _compound(text) == [
        "The formula of water is H2O",
        "each molecule contains three atoms.",
    ]


def test_semicolon_split():
    assert _compound("Paris is in France; Berlin is in Germany.") == [
        "Paris is in France",
        "Berlin is in Germany.",
    ]


def test_coordinated_noun_phrase_not_split():
    text = "The flag is red, white, and blue."
    assert _compound(text) == [text]


def test_right_side_without_verb_is_rejoined():
    text = "The museum holds paintings, and many old sculptures."
    assert _compound(text) == [text]
