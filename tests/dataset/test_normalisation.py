from scripts.dataset import normalisation

import pytest

@pytest.mark.parametrize(
    argnames=["before", "expected"],
    argvalues=[("typing.List[typing.List[typing.Tuple[int]]]", "typing.List[typing.List[Any]]")]
)
def test_to_limited(before: str, expected: str):
    normalised = normalisation.to_limited(before)

    assert expected == normalised, f"{expected=} != {normalised=}"


@pytest.mark.parametrize(
    argnames=["before", "expected"],
    argvalues=[("typing.List[typing.List[Any]]", "List[List]")]
)
def test_to_adjusted(before: str, expected: str):
    normalised = normalisation.to_adjusted(before)
    assert expected == normalised, f"{expected=} != {normalised=}"


@pytest.mark.parametrize(
    argnames=["before", "expected"],
    argvalues=[("typing.List[typing.List[Any]]", "List")]
)
def test_to_base(before: str, expected: str):
    normalised = normalisation.to_base(before)
    assert expected == normalised, f"{expected=} != {normalised=}"