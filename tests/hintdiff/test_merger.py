import pathlib
import typing

from common.schemas import (
    TypeCollectionCategory as TCCategory,
    TypeCollectionSchemaColumns,
    TypeCollectionSchema,
    MergedAnnotationSchema,
    MergedAnnotationSchemaColumns,
)
from common.storage import MergedAnnotations

import pandas as pd
from pandas._libs import missing
import pandera.typing as pt


import pytest

from hintdiff import cli


def resources() -> pathlib.Path:
    return pathlib.Path("tests", "resources")


@pytest.fixture
def symbol_diff() -> pd.DataFrame:
    # Only missing symbols are in proj2/x.py

    clazz_function = [
        ("x.py", TCCategory.CALLABLE_PARAMETER, "Clazz.function.a", "a.A", missing.NA),
        ("x.py", TCCategory.CALLABLE_PARAMETER, "Clazz.function.b", "b.B", missing.NA),
        ("x.py", TCCategory.CALLABLE_PARAMETER, "Clazz.function.c", "c.C", missing.NA),
        ("x.py", TCCategory.CALLABLE_RETURN, "Clazz.function", "int", missing.NA),
    ]

    a = [("x.py", TCCategory.VARIABLE, "a", "int", missing.NA)]

    return pd.DataFrame(
        clazz_function + a,
        columns=["file", "category", "qname", "proj1_anno", "proj2_anno"],
    )


# File order shall not matter (outer merge should make this work)
@pytest.mark.parametrize(
    argnames=["paths"],
    argvalues=[
        (tuple([resources() / "proj1", resources() / "proj2"]),),
        (tuple([resources() / "proj2", resources() / "proj1"]),),
    ],
    ids=["forward", "backward"],
)
def test_missing_symbols_exist(
    paths: tuple[pathlib.Path, pathlib.Path],
    symbol_diff: pd.DataFrame,
) -> None:
    merged = cli._collect(paths)

    # Columns called proj1_anno and proj2_anno must exist, containing the annotation for each symbol
    # in their respective files
    assert "proj1_anno" in merged.df.columns
    assert "proj2_anno" in merged.df.columns

    # Query entire accumulated dataset
    merge_diff = merged.differing()
    diff = pd.concat([merge_diff[symbol_diff.columns], symbol_diff]).drop_duplicates(keep=False)
    assert diff.empty

    # Query specific files
    merge_diff = merged.differing(files=[pathlib.Path("x.py")])
    diff = pd.concat([merge_diff[symbol_diff.columns], symbol_diff]).drop_duplicates(keep=False)
    assert diff.empty

    # Query specific projects
    merge_diff = merged.differing(roots=list(paths))
    diff = pd.concat([merge_diff[symbol_diff.columns], symbol_diff]).drop_duplicates(keep=False)
    assert diff.empty
