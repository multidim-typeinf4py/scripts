import pathlib
import typing

from common.storage import Schema, SchemaColumns, Category
from symbols import cli

import pandas as pd
import pandera as pa
import pandera.typing as pt

import pytest


@pytest.fixture
def code_path() -> typing.Iterator[pathlib.Path]:
    path = pathlib.Path("tests", "symbols", "x.py")
    content = path.open().read()

    yield path

    with path.open("w") as f:
        f.write(content)


@pytest.mark.parametrize(
    argnames=[
        "category",
        "hinted_symbols",
    ],
    argvalues=[
        (
            Category.CALLABLE_RETURN,
            [
                ("function", "int"),
                ("function_with_multiline_parameters", "int"),
                ("Clazz.__init__", "None"),
                ("Clazz.method", "tuple"),
                ("Clazz.multiline_method", "tuple"),
                ("Clazz.function", "int"),
                ("outer", "int"),
                ("outer.nested", "str"),
            ],
        ),
        (
            Category.CALLABLE_PARAMETER,
            [
                # function
                ("function.a", "int"),
                ("function.b", "str"),
                ("function.c", "int"),
                # function_with_multiline_parameters
                ("function_with_multiline_parameters.a", "str"),
                ("function_with_multiline_parameters.b", "int"),
                ("function_with_multiline_parameters.c", "str"),
                # Clazz.__init__
                ("Clazz.__init__.a", "int"),
                # Clazz.method
                ("Clazz.method.a", "int"),
                ("Clazz.method.b", "str"),
                ("Clazz.method.c", "int"),
                # Clazz.multiline_method
                ("Clazz.multiline_method.a", "str"),
                ("Clazz.multiline_method.b", "int"),
                ("Clazz.multiline_method.c", "str"),
                # Clazz.function
                ("Clazz.function.a", "a.A"),
                ("Clazz.function.b", "b.B"),
                ("Clazz.function.c", "c.C"),
                # outer.nested
                ("outer.nested.a", "int"),
            ],
        ),
    ],
    ids=[str(Category.CALLABLE_RETURN), str(Category.CALLABLE_PARAMETER)],
)
def test_returns(
    code_path: pathlib.Path, category: Category, hinted_symbols: list[tuple[str, str]]
) -> None:
    codemod_res, collection = cli._impl(code_path)
    assert codemod_res.failures == 0

    hints = [
        (str(code_path.name), category, qname, anno) for qname, anno in hinted_symbols
    ]
    hints_df: pt.DataFrame[Schema] = pd.DataFrame(hints, columns=SchemaColumns).pipe(
        pt.DataFrame[Schema]
    )

    # print("Expected: ", returns_df, sep="\n")
    # print("Actual: ", df, sep="\n")

    joined = pd.merge(collection._df, hints_df, on=SchemaColumns)
    assert len(joined) == len(
        hints_df
    ), f"Missing:\n{pd.concat([hints_df, joined]).drop_duplicates(keep=False)}"
