import collections
import pathlib
import textwrap
import typing

import libcst
import pandas as pd
import pandera.typing as pt
import pytest
from libcst import codemod
from libcst import metadata
from pandas._libs import missing

from common.ast_helper import generate_qname_ssas_for_file
from common.schemas import (
    TypeCollectionSchema,
    TypeCollectionSchemaColumns,
    TypeCollectionCategory,
)
from common.storage import TypeCollection
from symbols.collector import TypeCollectorVisitor, build_type_collection


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
                ("function", "function", "builtins.int"),
                (
                    "function_with_multiline_parameters",
                    "function_with_multiline_parameters",
                    "builtins.int",
                ),
                ("Clazz.__init__", "Clazz.__init__", "None"),
                ("Clazz.method", "Clazz.method", missing.NA),
                ("Clazz.multiline_method", "Clazz.multiline_method", "builtins.tuple"),
                ("Clazz.function", "Clazz.function", "builtins.int"),
                ("outer", "outer", "builtins.int"),
                ("outer.nested", "outer.nested", "builtins.str"),
            ],
        ),
        (
            TypeCollectionCategory.CALLABLE_PARAMETER,
            [
                # function
                ("function.a", "function.a", "builtins.int"),
                ("function.b", "function.b", "builtins.str"),
                ("function.c", "function.c", "builtins.int"),
                # function_with_multiline_parameters
                (
                    "function_with_multiline_parameters.a",
                    "function_with_multiline_parameters.a",
                    "builtins.str",
                ),
                (
                    "function_with_multiline_parameters.b",
                    "function_with_multiline_parameters.b",
                    "builtins.int",
                ),
                (
                    "function_with_multiline_parameters.c",
                    "function_with_multiline_parameters.c",
                    "builtins.str",
                ),
                # Clazz.__init__
                ("Clazz.__init__.self", "Clazz.__init__.self", missing.NA),
                ("Clazz.__init__.a", "Clazz.__init__.a", "builtins.int"),
                # Clazz.method
                ("Clazz.method.self", "Clazz.method.self", missing.NA),
                ("Clazz.method.a", "Clazz.method.a", "builtins.int"),
                ("Clazz.method.b", "Clazz.method.b", "builtins.str"),
                ("Clazz.method.c", "Clazz.method.c", "builtins.int"),
                # Clazz.multiline_method
                (
                    "Clazz.multiline_method.self",
                    "Clazz.multiline_method.self",
                    missing.NA,
                ),
                ("Clazz.multiline_method.a", "Clazz.multiline_method.a", "builtins.str"),
                ("Clazz.multiline_method.b", "Clazz.multiline_method.b", "builtins.int"),
                ("Clazz.multiline_method.c", "Clazz.multiline_method.c", missing.NA),
                # Clazz.function
                ("Clazz.function.self", "Clazz.function.self", missing.NA),
                ("Clazz.function.a", "Clazz.function.a", "amod.A"),
                ("Clazz.function.b", "Clazz.function.b", "bmod.B"),
                ("Clazz.function.c", "Clazz.function.c", "cmod.C"),
                # outer.nested
                ("outer.nested.a", "outer.nested.a", "builtins.int"),
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
                ("Clazz.__init__.self.a", "Clazz.__init__.self.aλ1", "builtins.int"),
                ("Clazz.function.v", "Clazz.function.vλ1", missing.NA),
                ("a", "aλ1", "builtins.int"),
                ("outer.nested.result", "outer.nested.resultλ1", "builtins.str"),
            ],
        ),
        (
            TypeCollectionCategory.INSTANCE_ATTR,
            [("Clazz.a", "Clazz.a", "builtins.int")],
        ),
    ],
    ids=["CALLABLE_RETURN", "CALLABLE_PARAMETER", "VARIABLE", "INSTANCE_ATTR"],
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
    m = df[df["_merge"] != "both"]
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


CR = collections.namedtuple(
    typename="CR",
    field_names=[
        TypeCollectionSchema.category,
        TypeCollectionSchema.qname,
        TypeCollectionSchema.anno,
    ],
)


class AnnotationTracking(codemod.CodemodTest):
    def performTracking(self, code: str) -> pt.DataFrame[TypeCollectionSchema]:
        module = libcst.parse_module(textwrap.dedent(code))

        visitor = TypeCollectorVisitor.strict(
            context=codemod.CodemodContext(
                filename="x.py",
                metadata_manager=metadata.FullRepoManager(
                    repo_root_dir=".", paths=["x.py"], providers=[]
                ),
            ),
        )

        module.visit(visitor)
        return visitor.collection.df

    def assertMatchingAnnotating(
        self,
        actual: pt.DataFrame[TypeCollectionSchema],
        expected: list[CR],
    ) -> None:
        if expected:
            expected_df = (
                pd.DataFrame(expected, columns=expected[0]._fields)
                .assign(file="x.py")
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
            a: int = ...
            def __init__(self):
                self.x: int = 0
                default: str = self.x or "10"
                self.x = default"""
        )

        self.assertMatchingAnnotating(
            df,
            [
                CR(TypeCollectionCategory.VARIABLE, "a", missing.NA),
                CR(TypeCollectionCategory.VARIABLE, "a", "builtins.str"),
                CR(TypeCollectionCategory.CALLABLE_RETURN, "f", missing.NA),
                CR(
                    TypeCollectionCategory.CALLABLE_PARAMETER,
                    "f.a",
                    missing.NA,
                ),
                CR(
                    TypeCollectionCategory.CALLABLE_PARAMETER,
                    "f.b",
                    missing.NA,
                ),
                CR(
                    TypeCollectionCategory.CALLABLE_PARAMETER,
                    "f.c",
                    missing.NA,
                ),
                CR(TypeCollectionCategory.INSTANCE_ATTR, "C.a", "builtins.int"),
                CR(
                    TypeCollectionCategory.CALLABLE_RETURN,
                    "C.__init__",
                    missing.NA,
                ),
                CR(
                    TypeCollectionCategory.CALLABLE_PARAMETER,
                    "C.__init__.self",
                    missing.NA,
                ),
                CR(TypeCollectionCategory.VARIABLE, "C.__init__.self.x", "builtins.int"),
                CR(TypeCollectionCategory.VARIABLE, "C.__init__.default", "builtins.str"),
                CR(
                    TypeCollectionCategory.VARIABLE,
                    "C.__init__.self.x",
                    missing.NA,
                ),
            ],
        )


class Test_HintTracking(AnnotationTracking):
    def test_no_store_when_unused(self):
        df = self.performTracking("a: int")
        self.assertMatchingAnnotating(
            df,
            [],
        )

    def test_hinting_not_merged(self):
        df = self.performTracking(
            """
        a: int
        a = 5
        """
        )
        self.assertMatchingAnnotating(df, [CR(TypeCollectionCategory.VARIABLE, "a", missing.NA)])

    def test_hinting_overwrite(self):
        # Unlikely to happen, but check anyway :)
        df = self.performTracking(
            """
            a: int
            a: str = "Hello World"
            """
        )
        self.assertMatchingAnnotating(df, [CR(TypeCollectionCategory.VARIABLE, "a", "builtins.str")])

    def test_stub_file_hinting(self):
        df = self.performTracking(
            """
            import requests.models
            r: requests.models.Response = ...
            """
        )

        self.assertMatchingAnnotating(
            df, [CR(TypeCollectionCategory.VARIABLE, "r", "requests.models.Response")]
        )

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
                CR(TypeCollectionCategory.VARIABLE, "a", "builtins.int"),
                CR(TypeCollectionCategory.VARIABLE, "b", "builtins.list[builtins.str]"),
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
                CR(TypeCollectionCategory.VARIABLE, "a", "builtins.list[builtins.str]"),
                CR(TypeCollectionCategory.VARIABLE, "b", "builtins.int"),
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
                CR(TypeCollectionCategory.VARIABLE, "a", "builtins.int"),
                CR(TypeCollectionCategory.VARIABLE, "b", "builtins.str"),
                CR(TypeCollectionCategory.VARIABLE, "d", "builtins.tuple[builtins.int, builtins.str]"),
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
                CR(TypeCollectionCategory.VARIABLE, "d", missing.NA),
                CR(TypeCollectionCategory.VARIABLE, "a", missing.NA),
                CR(TypeCollectionCategory.VARIABLE, "b", missing.NA),
                CR(TypeCollectionCategory.VARIABLE, "c", "builtins.int"),
            ],
        )

    def test_multiple_reassign(self):
        df = self.performTracking(
            """
            a: int = 5

            a: bytes
            a: str = "Hello World"

            a = 5
            """
        )
        self.assertMatchingAnnotating(
            df,
            [
                CR(TypeCollectionCategory.VARIABLE, "a", "builtins.int"),
                CR(TypeCollectionCategory.VARIABLE, "a", "builtins.str"),
                CR(TypeCollectionCategory.VARIABLE, "a", missing.NA),
            ],
        )

    def test_hint_retainment(self):
        df = self.performTracking(
            """
            a: int = 10
            a = 5

            a: str = "Hello"
            a = "World"
            """
        )
        self.assertMatchingAnnotating(
            df,
            [
                CR(TypeCollectionCategory.VARIABLE, "a", "builtins.int"),
                CR(TypeCollectionCategory.VARIABLE, "a", missing.NA),
                CR(TypeCollectionCategory.VARIABLE, "a", "builtins.str"),
                CR(TypeCollectionCategory.VARIABLE, "a", missing.NA),
            ],
        )

    @pytest.mark.skip(reason="Annotating NamedExprs is complicated!")
    def test_walrus(self):
        unannotated_df = self.performTracking(
            """
            (x := 4)
            """
        )
        self.assertMatchingAnnotating(
            unannotated_df, [CR(TypeCollectionCategory.VARIABLE, "x", missing.NA)]
        )

        annotated_df = self.performTracking(
            """
            x: int
            (x := 4)
            """
        )
        self.assertMatchingAnnotating(
            annotated_df, [CR(TypeCollectionCategory.VARIABLE, "x", "builtins.int")]
        )

    def test_for_unannotated(self):
        df = self.performTracking(
            """
            for x in [1, 2, 3]:
                ...
            """
        )
        self.assertMatchingAnnotating(df, [CR(TypeCollectionCategory.VARIABLE, "x", missing.NA)])

    def test_for_annotated(self):
        df = self.performTracking(
            """
            x: int
            for x in [1, 2, 3]:
                ...
            """
        )
        self.assertMatchingAnnotating(df, [CR(TypeCollectionCategory.VARIABLE, "x", "builtins.int")])

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
                CR(TypeCollectionCategory.VARIABLE, "x", "builtins.int"),
                CR(TypeCollectionCategory.VARIABLE, "y", "builtins.str"),
            ],
        )

    def test_withitem_unannotated(self):
        df = self.performTracking(
            """
        with open(file) as f:
            ...
        """
        )

        self.assertMatchingAnnotating(df, [CR(TypeCollectionCategory.VARIABLE, "f", missing.NA)])

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
            df, [CR(TypeCollectionCategory.VARIABLE, "f", "_io.TextIOWrapper")]
        )

    @pytest.mark.skip(reason="Cannot annotate comprehension loops, as their scope does not leak")
    def test_comprehension(self):
        df = self.performTracking(
            """
        [[x.value] for x in z]
        """
        )

        self.assertMatchingAnnotating(
            df,
            [
                CR(TypeCollectionCategory.VARIABLE, "x", missing.NA),
            ],
        )

    @pytest.mark.skip(reason="Cannot annotate comprehension loops, as their scope does not leak")
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
                CR(TypeCollectionCategory.VARIABLE, "x", "enum.Enum"),
            ],
        )

    def test_libsa4py(self):
        df = self.performTracking(
            """
            class C:
                foo = ...
                foo2: int = ...
            """
        )
        self.assertMatchingAnnotating(
            df,
            [
                CR(TypeCollectionCategory.INSTANCE_ATTR, "C.foo", missing.NA),
                CR(TypeCollectionCategory.INSTANCE_ATTR, "C.foo2", "builtins.int"),
            ],
        )

    def test_class_attribute(self):
        df = self.performTracking(
            """
            class C:
                a: int = 5
            """
        )
        self.assertMatchingAnnotating(
            df,
            [
                CR(TypeCollectionCategory.VARIABLE, "C.a", "builtins.int"),
            ],
        )

    def test_branch_annotating(self):
        df = self.performTracking(
            """
            import _io

            f: _io.TextWrapper
            a: int | None

            if cond:
                a = 1
                with p.open() as f:
                    ...
            else:
                a = None
                with q.open() as f:
                    ...
            """
        )
        self.assertMatchingAnnotating(
            df,
            [
                CR(TypeCollectionCategory.VARIABLE, "a", missing.NA),
                CR(TypeCollectionCategory.VARIABLE, "a", missing.NA),
                CR(TypeCollectionCategory.VARIABLE, "f", missing.NA),
                CR(TypeCollectionCategory.VARIABLE, "f", missing.NA),
            ],
        )

    def test_annotation_qualification(self):
        df = self.performTracking(
            """
            from typing import Callable
            import amod

            a: int = 5
            b: amod.B = amod.B(10)
            c: Callable = lambda: _
            # d: notimported.buthereanyway = 10
            """
        )
        self.assertMatchingAnnotating(
            df,
            [
                CR(TypeCollectionCategory.VARIABLE, "a", "builtins.int"),
                CR(TypeCollectionCategory.VARIABLE, "b", "amod.B"),
                CR(TypeCollectionCategory.VARIABLE, "c", "typing.Callable"),
                # CR(TypeCollectionCategory.VARIABLE, "d", "notimported.buthereanyway"),
            ],
        )

    def test_multi_import_handling(self):
        df = self.performTracking(
            """
            from typing import Callable
            import typing

            a: Callable = ...
            b: typing.Callable = ...
            """
        )
        self.assertMatchingAnnotating(
            df,
            [
                CR(TypeCollectionCategory.VARIABLE, "a", "typing.Callable"),
                CR(TypeCollectionCategory.VARIABLE, "b", "typing.Callable"),
            ],
        )

    def test_multi_assignment_class(self):
        df = self.performTracking(
            """
            class UserAdminView(AuthModelMixin):
                column_display_pk = True
                # Don't display the password on the list of Users
                column_exclude_list = list = ("password",)
                column_default_sort = ("created_at", True)
            """
        )
        self.assertMatchingAnnotating(
            df,
            [
                CR(TypeCollectionCategory.VARIABLE, "UserAdminView.column_display_pk", missing.NA),
                CR(
                    TypeCollectionCategory.VARIABLE, "UserAdminView.column_exclude_list", missing.NA
                ),
                CR(TypeCollectionCategory.VARIABLE, "UserAdminView.list", missing.NA),
                CR(
                    TypeCollectionCategory.VARIABLE, "UserAdminView.column_default_sort", missing.NA
                ),
            ],
        )
