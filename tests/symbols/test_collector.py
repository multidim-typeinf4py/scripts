import pathlib
import typing

from common.storage import Schema, SchemaColumns, Category
from symbols import cli

from pandas._libs import missing

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
                ("Clazz.method", missing.NA),
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
                ("Clazz.__init__.self", missing.NA),
                ("Clazz.__init__.a", "int"),
                # Clazz.method
                ("Clazz.method.self", missing.NA),
                ("Clazz.method.a", "int"),
                ("Clazz.method.b", "str"),
                ("Clazz.method.c", "int"),
                # Clazz.multiline_method
                ("Clazz.multiline_method.self", missing.NA),
                ("Clazz.multiline_method.a", "str"),
                ("Clazz.multiline_method.b", "int"),
                ("Clazz.multiline_method.c", missing.NA),
                # Clazz.function
                ("Clazz.function.self", missing.NA),
                ("Clazz.function.a", "a.A"),
                ("Clazz.function.b", "b.B"),
                ("Clazz.function.c", "c.C"),
                # outer.nested
                ("outer.nested.a", "int"),
            ],
        ),
        (
            Category.VARIABLE,
            [
                ("function.v", "str"),
                ("function_with_multiline_parameters.v", "str"),
                ("Clazz.__init__.self.a", "int"),
                ("Clazz.function.v", missing.NA),
                ("a", "int"),
                ("outer.nested.result", "str"),
            ],
        ),
        (
            Category.CLASS_ATTR,
            [
                ("Clazz.a", "int")
            ],
        ),
    ],
    ids=["CALLABLE_RETURN", "CALLABLE_PARAMETER", "VARIABLE", "CLASS_ATTR"],
)
def test_returns(
    code_path: pathlib.Path,
    category: Category,
    hinted_symbols: list[tuple[str, str | missing.NAType]],
) -> None:
    codemod_res, collection = cli._impl(code_path)
    assert codemod_res.failures == 0

    hints = [
        (str(code_path.name), category, qname, anno) for qname, anno in hinted_symbols
    ]
    hints_df: pt.DataFrame[Schema] = pd.DataFrame(hints, columns=SchemaColumns).pipe(
        pt.DataFrame[Schema]
    )

    print("Expected: ", hints_df, sep="\n")
    print("Actual: ", collection._df[collection._df["category"] == category], sep="\n")

    common = pd.merge(collection._df, hints_df, on=SchemaColumns)
    diff = pd.concat([common, hints_df]).drop_duplicates(keep=False)
    print("Diff:", diff, sep="\n")

    assert diff.empty
