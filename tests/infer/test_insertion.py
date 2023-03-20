import collections
import textwrap
import typing
import libcst

import pandas as pd
import pandera.typing as pt
import pytest
from libcst import codemod, metadata
from pandas._libs import missing

from common.ast_helper import generate_qname_ssas_for_file
from common.schemas import TypeCollectionCategory, TypeCollectionSchema
from infer.insertion import TypeAnnotationApplierTransformer
from symbols.collector import TypeCollectorVisitor

CodemodAnnotation = collections.namedtuple(
    typename="CodemodAnnotation",
    field_names=[
        TypeCollectionSchema.category,
        TypeCollectionSchema.qname,
        TypeCollectionSchema.anno,
    ],
)


class AnnotationTesting(codemod.CodemodTest):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.maxDiff = None

    def assertBuildCodemod(
        self,
        before: str,
        after: str,
        annotations: typing.Sequence[CodemodAnnotation] | None = None,
    ):
        before, after = textwrap.dedent(before), textwrap.dedent(after)

        context = codemod.CodemodContext(
            filename="x.py",
            metadata_manager=metadata.FullRepoManager(
                repo_root_dir=".", paths=["x.py"], providers=[]
            ),
        )

        if annotations is not None:
            df = (
                pd.DataFrame(annotations, columns=list(annotations[0]._fields))
                .assign(file="x.py")
                .pipe(generate_qname_ssas_for_file)
                .pipe(pt.DataFrame[TypeCollectionSchema])
            )

        else:
            visitor = TypeCollectorVisitor.strict(context)
            libcst.parse_module(after).visit(visitor)
            df = visitor.collection.df

        self.assertCodemod(
            before,
            after,
            annotations=df,
            context_override=context,
        )


class Test_Unannotated(AnnotationTesting):
    TRANSFORM = TypeAnnotationApplierTransformer

    def test_single_assign_variable_target(self):
        self.assertBuildCodemod(
            before="""
            a = 10
            """,
            after="""
            from __future__ import annotations

            a: int = 10
            """,
        )

    def test_instance_attribute_target(self):
        self.assertBuildCodemod(
            before="""
            class C:
                a = ...
            """,
            after="""
            from __future__ import annotations
            
            class C:
                a: int = ...
            """,
        )

    def test_class_attribute_target(self):
        self.assertBuildCodemod(
            before="""
            class C:
                a = 5
            """,
            after="""
            from __future__ import annotations
            
            class C:
                a: int = 5
            """,
        )

    def test_for_target(self):
        self.assertBuildCodemod(
            before="""
            for indexi, valuei in enumerate("Hello World"):
                for indexj, valuej in enumerate([[1, 2, 3]]):
                    ...
            """,
            after="""
            from __future__ import annotations

            indexi: int; valuei: str

            for indexi, valuei in enumerate("Hello World"):
                indexj: int; valuej: list
                for indexj, valuej in enumerate([[1, 2, 3]]):
                    ...
            """,
        )

    def test_multiple_assign_variable_target(self):
        self.assertBuildCodemod(
            before="""
            a = b = 50
            c, d = "HE"
            """,
            after="""
            from __future__ import annotations
            
            a: int; b: int; a = b = 50
            c: str; d: str; c, d = "HE"
            """,
        )

        self.assertBuildCodemod(
            before="""
            a = 10
            b, _ = 10, None
            c += "Hello"
            """,
            after="""
            from __future__ import annotations

            a: int = 10
            b: int; b, _ = 10, None
            c: str; c += "Hello"
            """,
        )

    def test_with_items(self):
        self.assertBuildCodemod(
            before="""
            import _io
            import scratchpad
            with make_scratchpad(path) as s, open(file) as f:
                ...
            """,
            after="""
            from __future__ import annotations

            import _io
            import scratchpad
            s: scratchpad.ScratchPad; f: _io.TextFileWrapper

            with make_scratchpad(path) as s, open(file) as f:
                ...
            """,
        )

    def test_global_stays_unannotated(self):
        self.assertBuildCodemod(
            before="""
            a: int = 0

            def f():
                global a
                a = 20
            """,
            after="""
            from __future__ import annotations

            a: int = 0

            def f():
                global a
                a = 20
            """,
        )

    def test_nonlocal_stays_unannotated(self):
        self.assertBuildCodemod(
            before="""
            def f():
                a: int = 10

                def g():
                    nonlocal a
                    a = 20
            """,
            after="""
            from __future__ import annotations

            def f():
                a: int = 10

                def g():
                    nonlocal a
                    a = 20
            """,
        )

    def test_parameters(self):
        self.assertBuildCodemod(
            before="""
            def f(a, b):
                ...
            """,
            after="""
            from __future__ import annotations

            def f(a: A, b: B):
                ...
            """,
        )

    def test_rettype(self):
        self.assertBuildCodemod(
            before="""
            def g(a, b):
                ...
            """,
            after="""
            from __future__ import annotations

            def g(a, b) -> int:
                ...
            """,
        )

    def test_libsa4py(self):
        self.assertBuildCodemod(
            before="""
            class C:
                a = ...
            """,
            after="""
            from __future__ import annotations

            class C:
                a: int = ...
            """,
        )

    def test_skip_unannotated(self):
        self.assertBuildCodemod(
            before=f"""
            from __future__ import annotations

            a = 10
            a = "Hello World"

            (b, c) = "Hello", 5

            def f(a, b, c): ...
            
            class C:
                a = ...
                def __init__(self):
                    self.x = 0
                    default = self.x or "10"
                    self.x = default
            """,
            after="""
            from __future__ import annotations

            a = 10
            a: str = "Hello World"

            b: str; (b, c) = "Hello", 5

            def f(a, b, c): ...
            
            class C:
                a = ...
                def __init__(self):
                    self.x: int = 0
                    default: str = self.x or "10"
                    self.x = default
            """,
        )

    @pytest.mark.skip(reason="Annotating NamedExprs is complicated!")
    def test_walrus(self):
        self.assertBuildCodemod(
            "(x := 4)",
            """
            from __future__ import annotations

            x: int
            (x := 4)
            """,
        )


class Test_Annotated(AnnotationTesting):
    TRANSFORM = TypeAnnotationApplierTransformer

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

    def test_function(self):
        self.assertBuildCodemod(
            before="""
            def f() -> int:
                return 42
            """,
            after="""
            from __future__ import annotations

            def f() -> int:
                return 42
            """,
            annotations=[
                CodemodAnnotation(TypeCollectionCategory.CALLABLE_RETURN, "f", "str")
            ],
        )

    def test_parameter(self):
        self.assertBuildCodemod(
            before="""
            def f(a: int, b):
                return a
            """,
            after="""
            from __future__ import annotations

            def f(a: int, b: int):
                return a
            """,
            annotations=[
                CodemodAnnotation(
                    TypeCollectionCategory.CALLABLE_RETURN, "f", missing.NA
                ),
                CodemodAnnotation(
                    TypeCollectionCategory.CALLABLE_PARAMETER, "f.a", "bool"
                ),
                CodemodAnnotation(
                    TypeCollectionCategory.CALLABLE_PARAMETER, "f.b", "int"
                ),
            ],
        )

    def test_explicitly_annotated(self):
        self.assertBuildCodemod(
            before="a: int = 5",
            after="""
            from __future__ import annotations

            a: int = 5
            """,
            annotations=[
                CodemodAnnotation(TypeCollectionCategory.VARIABLE, "a", "str")
            ],
        )

    def test_implictly_annotated(self):
        self.assertBuildCodemod(
            before="""
            a: int
            a = 5
            """,
            after="""
            from __future__ import annotations

            a: int
            a: str = 5
            """,
            annotations=[
                CodemodAnnotation(TypeCollectionCategory.VARIABLE, "a", "str")
            ],
        )

    def test_lowering_effects(self):
        self.assertBuildCodemod(
            before="""
            a: int | None
            if cond:
                a = 10
            else:
                a = None

            a = a or 30
            """,
            after="""
            from __future__ import annotations

            a: int | None
            if cond:
                a = 10
            else:
                a: None = None

            a = a or 30
            """,
        )

        self.assertBuildCodemod(
            before="""
            import _io

            f: _io.TextWrapper
            if cond:
                with p.open() as f:
                    ...
            else:
                with q.open() as f:
                    ...
            """,
            after="""
            from __future__ import annotations

            import _io

            f: _io.TextWrapper
            if cond:
                with p.open() as f:
                    ...
            else:
                with q.open() as f:
                    ...
            """,
        )
