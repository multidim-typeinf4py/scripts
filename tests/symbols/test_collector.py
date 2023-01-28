import pathlib
import textwrap
import typing

import libcst
from libcst import codemod
from libcst import metadata

from common.ast_helper import generate_qname_ssas_for_file
from common.schemas import (
    TypeCollectionSchema,
    TypeCollectionSchemaColumns,
    TypeCollectionCategory,
)
from common.storage import TypeCollection


from symbols.collector import TypeCollectorVistor, build_type_collection

from pandas._libs import missing

import pandas as pd
import pandera.typing as pt

import pytest


@pytest.fixture
def code_path() -> typing.Iterator[pathlib.Path]:
    path = pathlib.Path("tests", "resources", "proj1", "x.py")
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
            TypeCollectionCategory.CALLABLE_RETURN,
            [
                ("function", "function", "int"),
                ("function_with_multiline_parameters", "function_with_multiline_parameters", "int"),
                ("Clazz.__init__", "Clazz.__init__", "None"),
                ("Clazz.method", "Clazz.method", missing.NA),
                ("Clazz.multiline_method", "Clazz.multiline_method", "tuple"),
                ("Clazz.function", "Clazz.function", "int"),
                ("outer", "outer", "int"),
                ("outer.nested", "outer.nested", "str"),
            ],
        ),
        (
            TypeCollectionCategory.CALLABLE_PARAMETER,
            [
                # function
                ("function.a", "function.a", "int"),
                ("function.b", "function.b", "str"),
                ("function.c", "function.c", "int"),
                # function_with_multiline_parameters
                (
                    "function_with_multiline_parameters.a",
                    "function_with_multiline_parameters.a",
                    "str",
                ),
                (
                    "function_with_multiline_parameters.b",
                    "function_with_multiline_parameters.b",
                    "int",
                ),
                (
                    "function_with_multiline_parameters.c",
                    "function_with_multiline_parameters.c",
                    "str",
                ),
                # Clazz.__init__
                ("Clazz.__init__.self", "Clazz.__init__.self", missing.NA),
                ("Clazz.__init__.a", "Clazz.__init__.a", "int"),
                # Clazz.method
                ("Clazz.method.self", "Clazz.method.self", missing.NA),
                ("Clazz.method.a", "Clazz.method.a", "int"),
                ("Clazz.method.b", "Clazz.method.b", "str"),
                ("Clazz.method.c", "Clazz.method.c", "int"),
                # Clazz.multiline_method
                ("Clazz.multiline_method.self", "Clazz.multiline_method.self", missing.NA),
                ("Clazz.multiline_method.a", "Clazz.multiline_method.a", "str"),
                ("Clazz.multiline_method.b", "Clazz.multiline_method.b", "int"),
                ("Clazz.multiline_method.c", "Clazz.multiline_method.c", missing.NA),
                # Clazz.function
                ("Clazz.function.self", "Clazz.function.self", missing.NA),
                ("Clazz.function.a", "Clazz.function.a", "amod.A"),
                ("Clazz.function.b", "Clazz.function.b", "bmod.B"),
                ("Clazz.function.c", "Clazz.function.c", "cmod.C"),
                # outer.nested
                ("outer.nested.a", "outer.nested.a", "int"),
            ],
        ),
        (
            TypeCollectionCategory.VARIABLE,
            [
                ("function.v", "function.vλ1", missing.NA),
                (
                    "function_with_multiline_parameters.v",
                    "function_with_multiline_parameters.vλ1",
                    missing.NA,
                ),
                ("Clazz.__init__.self.a", "Clazz.__init__.self.aλ1", "int"),
                ("Clazz.function.v", "Clazz.function.vλ1", missing.NA),
                ("a", "aλ1", "int"),
                ("outer.nested.result", "outer.nested.resultλ1", "str"),
            ],
        ),
        (
            TypeCollectionCategory.CLASS_ATTR,
            [("Clazz.a", "Clazz.a", "int")],
        ),
    ],
    ids=["CALLABLE_RETURN", "CALLABLE_PARAMETER", "VARIABLE", "CLASS_ATTR"],
)
def test_hints_found(
    code_path: pathlib.Path,
    category: TypeCollectionCategory,
    hinted_symbols: list[tuple[str, str | missing.NAType]],
) -> None:
    collection = build_type_collection(code_path)

    hints = [
        (str(code_path.name), category, qname, qname_ssa, anno)
        for qname, qname_ssa, anno in hinted_symbols
    ]
    hints_df: pt.DataFrame[TypeCollectionSchema] = pd.DataFrame(
        hints, columns=TypeCollectionSchemaColumns
    ).pipe(pt.DataFrame[TypeCollectionSchema])

    print("Expected: ", hints_df, sep="\n")
    print(
        "Actual: ",
        collection.df[collection.df[TypeCollectionSchema.category] == category],
        sep="\n",
    )

    common = pd.merge(collection.df, hints_df, on=TypeCollectionSchemaColumns)
    diff = pd.concat([common, hints_df]).drop_duplicates(keep=False)

    assert diff.empty, f"Diff:\n{diff}\n"


def test_loadable(code_path: pathlib.Path) -> None:
    import tempfile

    collection = build_type_collection(code_path)
    with tempfile.NamedTemporaryFile(mode="w+") as tmpfile:
        collection.write(tmpfile.name)
        reloaded = TypeCollection.load(tmpfile.name)

        diff = pd.concat([collection.df, reloaded.df]).drop_duplicates(keep=False)
        print("Diff between in-memory and serde'd", diff, sep="\n")
        assert diff.empty


class Test_TrackUnannotated(codemod.CodemodTest):
    def test_unannotated_present(self):
        code = textwrap.dedent(
            """
        from __future__ import annotations

        a = 10
        a: str = "Hello World"

        def f(a, b, c): ...
        
        class C:
            def __init__(self):
                self.x: int = 0
                default: str = self.x or "10"
                self.x = default"""
        )
        module = libcst.parse_module(code)

        expected_df = (
            pd.DataFrame(
                {
                    "file": ["x.py"] * 5,
                    "category": [TypeCollectionCategory.VARIABLE] * 5,
                    "qname": ["a"] * 2
                    + [f"C.__init__.{v}" for v in ("self.x", "default", "self.x")],
                    "anno": [missing.NA, "str", "int", "str", missing.NA],
                }
            )
            .pipe(generate_qname_ssas_for_file)
            .pipe(pt.DataFrame[TypeCollectionSchema])
        )

        visitor = TypeCollectorVistor.strict(
            context=codemod.CodemodContext(
                filename="x.py",
                metadata_manager=metadata.FullRepoManager(
                    repo_root_dir=".", paths=["x.py"], providers=[]
                ),
            ),
        )
        visitor.transform_module(module)

        df = visitor.collection.df

        common = pd.merge(expected_df, df)
        diff = pd.concat([common, expected_df]).drop_duplicates(keep=False)

        assert diff.empty, f"Diff:\n{diff}\n"
