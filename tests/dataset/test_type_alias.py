from scripts.dataset.type_alias import is_type_alias

import pytest


def test_missing_annotation_is_not_type_alias() -> None:
    assert is_type_alias(None) is False


@pytest.mark.parametrize(
    argnames="alias",
    argvalues=[
        "typing.Type",
        "Type",
        "typing.Type[int]",
        "Type[int]",
    ],
)
def test_typing_type_is_type_alias(alias: str) -> None:
    assert is_type_alias(alias)


@pytest.mark.parametrize(
    argnames="alias",
    argvalues=[
        "typing.TypeAlias",
        "TypeAlias",
    ],
)
def test_typing_type_is_type_alias(alias: str) -> None:
    assert is_type_alias(alias)


@pytest.mark.parametrize(
    argnames="alias",
    argvalues=[
        "typing.TypeVar",
        "TypeVar",
    ],
)
def test_typing_typevar_is_type_alias(alias: str) -> None:
    assert is_type_alias(alias)


@pytest.mark.parametrize(
    argnames="alias",
    argvalues=[
        "typing.NewType",
        "NewType",
    ],
)
def test_newtype_is_type_alias(alias: str) -> None:
    assert is_type_alias(alias)


@pytest.mark.parametrize(
    argnames="alias",
    argvalues=[
        "builtins.type",
        "builtins.type[int]",
        "type",
        "type[int]",
    ],
)
def test_builtin_type_is_type_alias(alias: str) -> None:
    assert is_type_alias(alias)
