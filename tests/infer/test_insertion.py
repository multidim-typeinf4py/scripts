import textwrap


from libcst import codemod, metadata
import pytest

from common.ast_helper import generate_qname_ssas_for_file
from common.schemas import TypeCollectionCategory, TypeCollectionSchema

from infer.insertion import TypeAnnotationApplierTransformer

import pandas as pd
from pandas._libs import missing
import pandera.typing as pt


class AnnotationTesting(codemod.CodemodTest):
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


class Test_CustomAnnotator(AnnotationTesting):
    TRANSFORM = TypeAnnotationApplierTransformer

    def assertBuildCodemod(self, before: str, after: str, annotations: pd.DataFrame):
        before = textwrap.dedent(before)
        after = textwrap.dedent(after)

        self.assertCodemod(
            before,
            after,
            annotations=annotations.assign(file="x.py")
            .pipe(generate_qname_ssas_for_file)
            .pipe(pt.DataFrame[TypeCollectionSchema]),
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
            
            import typing

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
            annotations=pd.DataFrame(
                {
                    "category": [TypeCollectionCategory.VARIABLE] * 7,
                    "qname": ["a"] * 2
                    + ["b", "c"]
                    + [f"C.__init__.{v}" for v in ("self.x", "default", "self.x")],
                    "anno": ["int", "str", "str", "int", "int", "str", "str"],
                }
            ),
        )

    def test_skip_unannotated_variables(self):
        self.assertBuildCodemod(
            before=AnnotationTesting.HINTLESS,
            after=f"""
            from __future__ import annotations
            import typing
        
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
            import typing
            
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
            annotations=pd.DataFrame(
                {
                    "category": [TypeCollectionCategory.CALLABLE_PARAMETER] * 4,
                    "qname": [f"f.{v}" for v in "abc"] + ["C.__init__.self"],
                    "anno": ["amod.A", "bmod.B", "cmod.C", "C"],
                }
            ),
        )

    def test_rettype(self):
        self.assertBuildCodemod(
            before=AnnotationTesting.HINTLESS,
            after="""
            from __future__ import annotations
            import typing

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
            annotations=pd.DataFrame(
                {
                    "category": [TypeCollectionCategory.CALLABLE_RETURN] * 2,
                    "qname": ["f", "C.__init__"],
                    "anno": ["int", "None"],
                }
            ),
        )

    def test_instance_attribute(self):
        self.assertBuildCodemod(
            before=AnnotationTesting.HINTLESS,
            after="""
            from __future__ import annotations
            import typing

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
            annotations=pd.DataFrame(
                {
                    "category": [TypeCollectionCategory.INSTANCE_ATTR] * 1,
                    "qname": ["C.a"],
                    "anno": ["int"],
                }
            ),
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
            import typing

            a: int = 10
            b: int; (b, _) = 10, None
            c: str; c += "Hello"
            """,
            annotations=(
                pd.DataFrame(
                    {
                        "category": [TypeCollectionCategory.VARIABLE] * 3,
                        "qname": list("abc"),
                        "anno": ["int"] * 2 + ["str"],
                    }
                )
            ),
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
