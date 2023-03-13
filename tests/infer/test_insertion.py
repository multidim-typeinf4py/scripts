import textwrap


from libcst import codemod, metadata
import libcst
import pytest

from common.ast_helper import generate_qname_ssas_for_file
from common.schemas import TypeCollectionCategory, TypeCollectionSchema

from infer.insertion import TypeAnnotationApplierTransformer

import pandas as pd
from pandas._libs import missing
import pandera.typing as pt

from symbols.collector import TypeCollectorVisitor


class AnnotationTesting(codemod.CodemodTest):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.maxDiff = None

    HINTLESS = textwrap.dedent(
        """
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
        """
    )

    HINTED = textwrap.dedent(
        """
        a: int = 10
        a: str = "Hello World"

        b: str; c: int
        (b, c) = "Hello", 5

        def f(a: amod.A, b: bmod.B, c: cmod.C) -> int: ...
        
        class C:
            a: int = ...
            def __init__(self: "C") -> None:
                self.x: int = 0
                default: str = self.x or "10"
                self.x: str = default
    """
    )


class Test_CustomAnnotator(AnnotationTesting):
    TRANSFORM = TypeAnnotationApplierTransformer

    def assertBuildCodemod(
        self,
        before: str,
        after: str,
        annotations: pd.DataFrame | list[TypeCollectionCategory],
    ):
        visitor = TypeCollectorVisitor.strict(
            context=codemod.CodemodContext(
                filename="x.py",
                metadata_manager=metadata.FullRepoManager(
                    repo_root_dir=".", paths=["x.py"], providers=[]
                ),
            )
        )

        if isinstance(annotations, pd.DataFrame):
            annotations = (
                annotations.assign(file="x.py")
                .pipe(generate_qname_ssas_for_file)
                .pipe(pt.DataFrame[TypeCollectionSchema])
            )

        elif isinstance(annotations, list):
            libcst.parse_module(self.HINTED).visit(visitor)

            df = visitor.collection.df
            masked = ~df[TypeCollectionSchema.category].isin(annotations)
            df.loc[masked, TypeCollectionSchema.anno] = missing.NA

            annotations = df

        else:
            assert False, f"Unsupported {annotations=}"


        self.assertCodemod(
            before,
            after,
            annotations=annotations,
            context_override=codemod.CodemodContext(
                filename="x.py",
                metadata_manager=metadata.FullRepoManager(
                    repo_root_dir=".", paths=["x.py"], providers=[]
                ),
            ),
        )

    def test_attributes(self):
        self.assertBuildCodemod(
            before=AnnotationTesting.HINTLESS,
            after="""
            from __future__ import annotations

            a: int = 10
            a: str = "Hello World"

            b: str; c: int; (b, c) = "Hello", 5

            def f(a, b, c): ...
            
            class C:
                a = ...
                def __init__(self):
                    self.x: int = 0
                    default: str = self.x or "10"
                    self.x: str = default
            """,
            annotations=[TypeCollectionCategory.VARIABLE],
        )

    def test_skip_unannotated_variables(self):
        self.assertBuildCodemod(
            before=AnnotationTesting.HINTLESS,
            after=f"""
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
            annotations=pd.DataFrame(
                {
                    "category": [TypeCollectionCategory.VARIABLE] * 7,
                    "qname": ["a"] * 2
                    + ["b", "c"]
                    + [f"C.__init__.{v}" for v in ("self.x", "default", "self.x")],
                    "anno": [missing.NA, "str", "str", missing.NA, "int", "str", missing.NA],
                }
            ),
        )

    def test_parameters(self):
        self.assertBuildCodemod(
            before=AnnotationTesting.HINTLESS,
            after="""
            from __future__ import annotations

            a = 10
            a = "Hello World"

            (b, c) = "Hello", 5

            def f(a: amod.A, b: bmod.B, c: cmod.C): ...
            
            class C:
                a = ...
                def __init__(self: "C"):
                    self.x = 0
                    default = self.x or "10"
                    self.x = default
            """,
            annotations=[TypeCollectionCategory.CALLABLE_PARAMETER],
        )

    def test_rettype(self):
        self.assertBuildCodemod(
            before=AnnotationTesting.HINTLESS,
            after="""
            from __future__ import annotations

            a = 10
            a = "Hello World"

            (b, c) = "Hello", 5

            def f(a, b, c) -> int: ...
            
            class C:
                a = ...
                def __init__(self) -> None:
                    self.x = 0
                    default = self.x or "10"
                    self.x = default
            """,
            annotations=[TypeCollectionCategory.CALLABLE_RETURN],
        )

    def test_instance_attribute(self):
        self.assertBuildCodemod(
            before=AnnotationTesting.HINTLESS,
            after="""
            from __future__ import annotations

            a = 10
            a = "Hello World"

            (b, c) = "Hello", 5

            def f(a, b, c): ...
            
            class C:
                a: int = ...
                def __init__(self):
                    self.x = 0
                    default = self.x or "10"
                    self.x = default
            """,
            annotations=[TypeCollectionCategory.INSTANCE_ATTR],
        )

    def test_assign_hinting(self):
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
            annotations=(
                pd.DataFrame(
                    {
                        "category": [TypeCollectionCategory.VARIABLE] * 4,
                        "qname": list("ab_c"),
                        "anno": ["int"] * 2 + [missing.NA, "str"],
                    }
                )
            ),
        )

    def test_for_loop(self):
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
            annotations=pd.DataFrame({
                "category": [TypeCollectionCategory.VARIABLE] * 4,
                "qname": ["indexi", "valuei", "indexj", "valuej"],
                "anno": ["int", "str", "int", "list"]
            })
        )

    def test_with_items(self):
        self.assertBuildCodemod(
            before="""
            with scratchpad(path) as s, open(file) as f:
                ...
            """,
            after="""
            from __future__ import annotations

            s: scratchpad.ScratchPad; f: _io.TextFileWrapper

            with scratchpad(path) as s, open(file) as f:
                ...
            """,
            annotations=pd.DataFrame({
                "category": [TypeCollectionCategory.VARIABLE] * 2,
                "qname": ["s", "f"],
                "anno": ["scratchpad.ScratchPad", "_io.TextFileWrapper"]
            })
        )

    @pytest.mark.skip(reason="Annotating NamedExprs is complicated!")
    def test_walrus(self):
        self.assertBuildCodemod(
            "(x := 4)",
            """
            x: int
            (x := 4)
            """,
            annotations=pd.DataFrame(
                {
                    "category": [TypeCollectionCategory.VARIABLE],
                    "qname": ["x"],
                    "anno": ["int"],
                }
            ),
        )
