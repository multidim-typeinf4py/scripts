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


def test_returns(code_path: pathlib.Path) -> None:
    codemod_res, collection = cli._impl(code_path)
    assert codemod_res.failures == 0

    df = collection._df

    hints = [
        ("function", "int"),
        ("function_with_multiline_parameters", "int"),
        ("Clazz.__init__", "None"),
        ("Clazz.method", "tuple"),
        ("Clazz.multiline_method", "tuple"),
        ("Clazz.function", "int"),
        ("outer", "int"),
        ("outer.nested", "str"),
    ]

    returns = [
        (str(code_path), Category.CALLABLE_RETURN, qname, anno) for qname, anno in hints
    ]
    returns_df: pt.DataFrame[Schema] = pd.DataFrame(returns, columns=SchemaColumns).pipe(pt.DataFrame[Schema])

    m = pd.merge(df, returns_df, on=df.columns)
    assert len(m) == len(
        returns_df
    ), f"{pd.concat([returns_df,m]).drop_duplicates(keep=False)}"
