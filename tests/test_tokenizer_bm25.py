import math

import pytest

from hallucination_lab.retrieval.bm25 import BM25Index, BM25Params
from hallucination_lab.retrieval.tokenizer import tokenize


@pytest.mark.parametrize(
    "text,expected",
    [
        ("8,848.86 m", ["8848.86", "8848", "metre"]),
        ("9,848 metres", ["9848", "metre"]),
        ("73%", ["73", "percent"]),
        ("100 °C", ["100", "celsius"]),
        ("29,031 ft", ["29031", "foot"]),
        ("São Paulo\u2019s", ["sao", "paulo"]),
        ("The tower WAS completed", ["tower", "completed"]),
        ("cities countries", ["city", "country"]),
        ("Mars Paris species", ["mars", "paris", "species"]),
        ("Champ-de-Mars", ["champ", "de", "mars"]),
        ("1,2", ["1", "2"]),
    ],
)
def test_tokenize(text, expected):
    assert tokenize(text) == expected


def test_dates_and_years_survive_tokenization():
    assert tokenize("On 3 May 1921 and in 1874.") == ["3", "may", "1921", "1874"]


def test_stopword_only_text_has_no_tokens():
    assert tokenize("It is what it was, and so on.") == []


def test_query_and_document_spellings_meet():
    assert set(tokenize("meters")) == set(tokenize("metres"))


def test_bm25_matches_hand_computed_score():
    index = BM25Index([["a", "b"], ["a", "c", "c"]], BM25Params(k1=1.5, b=0.75))
    (top,) = [s for s in index.score(["c"])]
    idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    norm = 1.5 * (1 - 0.75 + 0.75 * 3 / 2.5)
    assert top.index == 1
    assert top.score == pytest.approx(idf * 2 * 2.5 / (2 + norm))
    assert top.term_contributions == {"c": pytest.approx(top.score)}


def test_idf_never_negative_for_common_terms():
    index = BM25Index([["x"], ["x"], ["x", "y"]])
    assert index.idf("x") > 0
    assert all(s.score > 0 for s in index.score(["x"]))


def test_ties_break_by_position_deterministically():
    index = BM25Index([["q", "z"], ["q", "z"], ["q", "z"]])
    assert [s.index for s in index.score(["q"])] == [0, 1, 2]


def test_unknown_terms_score_nothing():
    assert BM25Index([["a"]]).score(["zzz"]) == []


def test_empty_index_rejected():
    with pytest.raises(ValueError):
        BM25Index([])


@pytest.mark.parametrize("k1,b", [(-1, 0.5), (1.2, 1.5)])
def test_invalid_params(k1, b):
    with pytest.raises(ValueError):
        BM25Params(k1, b)
