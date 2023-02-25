import operator
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

    df = pd.merge(collection.df, hints_df, how="right", indicator=True)

    m = df[df["_merge"] == "right_only"]
    print(m)

    assert m.empty, f"Diff:\n{m}\n"


def test_loadable(code_path: pathlib.Path) -> None:
    import tempfile

    collection = build_type_collection(code_path)
    with tempfile.NamedTemporaryFile(mode="w+") as tmpfile:
        collection.write(tmpfile.name)
        reloaded = TypeCollection.load(tmpfile.name)

        diff = pd.concat([collection.df, reloaded.df]).drop_duplicates(keep=False)
        print("Diff between in-memory and serde'd", diff, sep="\n")
        assert diff.empty


class AnnotationTracking(codemod.CodemodTest):
    def performTracking(self, code: str) -> pt.DataFrame[TypeCollectionSchema]:
        module = libcst.parse_module(textwrap.dedent(code))

        visitor = TypeCollectorVistor.strict(
            context=codemod.CodemodContext(
                filename="x.py",
                metadata_manager=metadata.FullRepoManager(
                    repo_root_dir=".", paths=["x.py"], providers=[]
                ),
            ),
        )
        visitor.transform_module(module)

        return visitor.collection.df

    def assertMatchingAnnotating(
        self,
        actual: pt.DataFrame[TypeCollectionSchema],
        expected: list[tuple[TypeCollectionCategory, str, str | missing.NAType]],
    ) -> None:
        files = ["x.py"] * len(expected)
        categories = list(map(operator.itemgetter(0), expected))
        qnames = list(map(operator.itemgetter(1), expected))
        annos = list(map(operator.itemgetter(2), expected))

        if expected:
            expected_df = (
                pd.DataFrame(
                    {"file": files, "category": categories, "qname": qnames, "anno": annos}
                )
                .pipe(generate_qname_ssas_for_file)
                .pipe(pt.DataFrame[TypeCollectionSchema])
            )
        else:
            expected_df = TypeCollectionSchema.example(size=0)

        comparison = pd.merge(actual, expected_df, how="outer", indicator=True)
        print(comparison)

        m = comparison[comparison["_merge"] != "both"]
        # print(m)
        assert m.empty, f"Diff:\n{m}\n"


class Test_TrackUnannotated(AnnotationTracking):
    def test_unannotated_present(self):
        df = self.performTracking(
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

        self.assertMatchingAnnotating(
            df,
            [
                (TypeCollectionCategory.VARIABLE, "a", missing.NA),
                (TypeCollectionCategory.VARIABLE, "a", "str"),
                (TypeCollectionCategory.CALLABLE_RETURN, "f", missing.NA),
                (TypeCollectionCategory.CALLABLE_PARAMETER, "f.a", missing.NA),
                (TypeCollectionCategory.CALLABLE_PARAMETER, "f.b", missing.NA),
                (TypeCollectionCategory.CALLABLE_PARAMETER, "f.c", missing.NA),
                (TypeCollectionCategory.CALLABLE_RETURN, "C.__init__", missing.NA),
                (TypeCollectionCategory.CALLABLE_PARAMETER, "C.__init__.self", missing.NA),
                (TypeCollectionCategory.VARIABLE, "C.__init__.self.x", "int"),
                (TypeCollectionCategory.VARIABLE, "C.__init__.default", "str"),
                (TypeCollectionCategory.VARIABLE, "C.__init__.self.x", missing.NA),
            ],
        )


class Test_HintTracking(AnnotationTracking):
    def test_no_store_when_unused(self):
        df = self.performTracking("a: int")
        self.assertMatchingAnnotating(
            df,
            [],
        )

    def test_hinting_merged(self):
        df = self.performTracking(
            """
        a: int
        a = 5
        """
        )
        self.assertMatchingAnnotating(df, [(TypeCollectionCategory.VARIABLE, "a", "int")])

    def test_hinting_overwrite(self):
        # Unlikely to happen, but check anyway :)
        df = self.performTracking(
            """
            a: int
            a: str = "Hello World"
            """
        )
        self.assertMatchingAnnotating(df, [(TypeCollectionCategory.VARIABLE, "a", "str")])

    def test_hinting_applied_to_unpackables(self):
        tuple_df = self.performTracking(
            """
            a: int
            b: list[str]

            a, *b = 5, "Hello World"
            """
        )
        self.assertMatchingAnnotating(
            tuple_df,
            [
                (TypeCollectionCategory.VARIABLE, "a", "int"),
                (TypeCollectionCategory.VARIABLE, "b", "list[str]"),
            ],
        )

        list_df = self.performTracking(
            """
            a: list[str]
            b: int

            [*a, b] = "hello world", 10
            """
        )
        self.assertMatchingAnnotating(
            list_df,
            [
                (TypeCollectionCategory.VARIABLE, "a", "list[str]"),
                (TypeCollectionCategory.VARIABLE, "b", "int"),
            ],
        )

    def test_hinting_applied_to_chained_assignment(self):
        df = self.performTracking(
            """
        a: int
        b: str
        d: tuple[int, str]

        d = (a, b) = (5, "Test")
        """
        )

        self.assertMatchingAnnotating(
            df,
            [
                (TypeCollectionCategory.VARIABLE, "a", "int"),
                (TypeCollectionCategory.VARIABLE, "b", "str"),
                (TypeCollectionCategory.VARIABLE, "d", "tuple[int, str]"),
            ],
        )

    def test_deep_unpackable_recursion(self):
        df = self.performTracking(
            """
            d: int
            d = 5

            c: int
            a, (b, c) = 5, (10, 20)
            """
        )
        self.assertMatchingAnnotating(
            df,
            [
                (TypeCollectionCategory.VARIABLE, "a", missing.NA),
                (TypeCollectionCategory.VARIABLE, "b", missing.NA),
                (TypeCollectionCategory.VARIABLE, "c", "int"),
                (TypeCollectionCategory.VARIABLE, "d", "int"),
            ],
        )

    def test_multiple_reassign(self):
        df = self.performTracking(
            """
            a: int
            a = 5

            a: bytes
            a: str = "Hello World"

            a = 5
            """
        )
        self.assertMatchingAnnotating(
            df,
            [
                (TypeCollectionCategory.VARIABLE, "a", "int"),
                (TypeCollectionCategory.VARIABLE, "a", "str"),
                (TypeCollectionCategory.VARIABLE, "a", missing.NA),
            ],
        )

    def test_hint_retainment(self):
        df = self.performTracking(
            """
            a: int
            a = 10
            a = 5

            a: str
            a = "Hello"
            a = "World"
            """
        )
        self.assertMatchingAnnotating(
            df,
            [(TypeCollectionCategory.VARIABLE, "a", "int")] * 2
            + [(TypeCollectionCategory.VARIABLE, "a", "str")] * 2,
        )

    def test_walrus(self):
        unannotated_df = self.performTracking(
            """
            (x := 4)
            """
        )
        self.assertMatchingAnnotating(
            unannotated_df, [(TypeCollectionCategory.VARIABLE, "x", missing.NA)]
        )

        annotated_df = self.performTracking(
            """
            x: int
            (x := 4)
            """
        )
        self.assertMatchingAnnotating(annotated_df, [(TypeCollectionCategory.VARIABLE, "x", "int")])

    def test_for_unannotated(self):
        df = self.performTracking(
            """
            for x in [1, 2, 3]:
                ...
            """
        )
        self.assertMatchingAnnotating(df, [(TypeCollectionCategory.VARIABLE, "x", missing.NA)])

    def test_for_annotated(self):
        df = self.performTracking(
            """
            x: int
            for x in [1, 2, 3]:
                ...
            """
        )
        self.assertMatchingAnnotating(df, [(TypeCollectionCategory.VARIABLE, "x", "int")])

    def test_for_unpacking_annotated(self):
        df = self.performTracking(
            """
            x: int
            y: str

            for x, y in zip([1, 2, 3], "abc"):
                ...
            """
        )
        self.assertMatchingAnnotating(
            df,
            [
                (TypeCollectionCategory.VARIABLE, "x", "int"),
                (TypeCollectionCategory.VARIABLE, "y", "str"),
            ],
        )

    def test_withitem_unannotated(self):
        df = self.performTracking(
            """
        with open(file) as f:
            ...
        """
        )

        self.assertMatchingAnnotating(df, [(TypeCollectionCategory.VARIABLE, "f", missing.NA)])

    def test_withitem_annotated(self):
        df = self.performTracking(
            """
        import _io

        f: _io.TextIOWrapper
        with open(file) as f:
            ...
        """
        )

        self.assertMatchingAnnotating(
            df, [(TypeCollectionCategory.VARIABLE, "f", "_io.TextIOWrapper")]
        )

    def test_comprehension(self):
        df = self.performTracking(
            """
        [[x.value] for x in z]
        """
        )

        self.assertMatchingAnnotating(
            df,
            [
                (TypeCollectionCategory.VARIABLE, "x", missing.NA),
            ],
        )

    def test_comprehension_annotated(self):
        df = self.performTracking(
            """
        from enum import Enum

        x: Enum
        [[x.value] for x in z]
        """
        )

        self.assertMatchingAnnotating(
            df,
            [
                (TypeCollectionCategory.VARIABLE, "x", "enum.Enum"),
            ],
        )
