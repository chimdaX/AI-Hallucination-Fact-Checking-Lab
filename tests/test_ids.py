import pytest

from hallucination_lab.ids import (
    content_sha256,
    is_valid_id,
    make_id,
    random_id,
    response_id_for,
)


def test_make_id_is_deterministic_and_well_formed():
    a = make_id("clm", "rsp_0000000000000000", 0, 10)
    assert a == make_id("clm", "rsp_0000000000000000", 0, 10)
    assert is_valid_id(a)
    assert a.startswith("clm_")


def test_make_id_distinguishes_part_boundaries():
    assert make_id("clm", "ab", "c") != make_id("clm", "a", "bc")


def test_response_id_depends_on_exact_text():
    assert response_id_for("Hello world.") != response_id_for("Hello world. ")


@pytest.mark.parametrize("prefix", ["CLM", "cl", "clm1", ""])
def test_make_id_rejects_bad_prefix(prefix):
    with pytest.raises(ValueError):
        make_id(prefix, "x")


@pytest.mark.parametrize(
    "value", ["../etc/passwd", "ses_123", "ses_ZZZZZZZZZZZZZZZZ", "ses_0123456789abcdef.json"]
)
def test_is_valid_id_rejects_unsafe_values(value):
    assert not is_valid_id(value)


def test_random_ids_differ_but_are_valid():
    a, b = random_id("ses"), random_id("ses")
    assert a != b and is_valid_id(a) and is_valid_id(b)


def test_content_sha256_length():
    assert len(content_sha256("x")) == 64
