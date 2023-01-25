import textwrap


from libcst import codemod, metadata

from common._helper import generate_qname_ssas_for_file
from common.schemas import TypeCollectionCategory, TypeCollectionSchema

from icr.insertion import (
    QName2SSATransformer,
    SSA2QNameTransformer,
    TypeAnnotationApplierTransformer,
)

import pandas as pd
from pandas._libs import missing
import pandera.typing as pt

class AnnotationTesting(codemod.CodemodTest):
    HINTLESS = textwrap.dedent(
        """
        a = 10
        a = "Hello World"

        def f(a, b, c): ...
        
        class C:
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

        def f(a, b, c): ...
        
        class C:
            def __init__(self):
                self.x: int = 0
                default: str = self.x or "10"
                self.x: str = default
        """
    )

    HINTED_QNAME_SSA = textwrap.dedent(
        """
        aλ1: int = 10
        aλ2: str = "Hello World"

        def f(a, b, c): ...
        
        class C:
            def __init__(self):
                self.xλ1: int = 0
                defaultλ1: str = self.x or "10"
                self.xλ2: str = default
        """
    )

    ANNOS = (
        pd.DataFrame(
            {
                "file": ["x.py"] * 5,
                "category": [TypeCollectionCategory.VARIABLE] * 5,
                "qname": ["a"] * 2 + [f"C.__init__.{v}" for v in ("self.x", "default", "self.x")],
                "anno": ["int", "str", "int", "str", "str"],
            }
        )
        .pipe(generate_qname_ssas_for_file)
        .pipe(pt.DataFrame[TypeCollectionSchema])
    )

    FILENAME = ANNOS["file"].iloc[0]


class Test_QName2QNameSSA(AnnotationTesting):
    TRANSFORM = QName2SSATransformer

    def test_qnames_transformed(self):
        self.assertCodemod(
            AnnotationTesting.HINTED,
            AnnotationTesting.HINTED_QNAME_SSA,
            annotations=AnnotationTesting.ANNOS,
        )


class Test_QNameSSA2QName(AnnotationTesting):
    TRANSFORM = SSA2QNameTransformer

    def test_qname_ssas_transformed(self):
        self.assertCodemod(
            AnnotationTesting.HINTED_QNAME_SSA,
            AnnotationTesting.HINTED,
            annotations=AnnotationTesting.ANNOS,
        )


class Test_CustomAnnotator(AnnotationTesting):
    TRANSFORM = TypeAnnotationApplierTransformer

    def test_attributes(self):
        filename = AnnotationTesting.ANNOS[TypeCollectionSchema.file].iloc[0]

        with_future_import = f"from __future__ import annotations\nimport typing\nfrom typing import *\n{AnnotationTesting.HINTED}"

        self.assertCodemod(
            AnnotationTesting.HINTLESS,
            with_future_import,
            tycol=AnnotationTesting.ANNOS,
            context_override=codemod.CodemodContext(
                filename=filename,
                metadata_manager=metadata.FullRepoManager(
                    repo_root_dir=".", paths=[filename], providers=[]
                ),
            ),
        )

    def test_skip_unannotated_variables(self):
        after = textwrap.dedent(
            f"""
        from __future__ import annotations
        import typing
        from typing import *
    
        a = 10
        a: str = "Hello World"

        def f(a, b, c): ...
        
        class C:
            def __init__(self):
                self.x: int = 0
                default: str = self.x or "10"
                self.x = default
        """
        )

        skip_df = (
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

        self.assertCodemod(
            AnnotationTesting.HINTLESS,
            after,
            tycol=skip_df,
            context_override=codemod.CodemodContext(
                filename=AnnotationTesting.FILENAME,
                metadata_manager=metadata.FullRepoManager(
                    repo_root_dir=".", paths=[AnnotationTesting.FILENAME], providers=[]
                ),
            ),
        )

    def test_parameters(self):
        after = textwrap.dedent(
            f"""
        from __future__ import annotations
        import typing
        from typing import *
        
        a = 10
        a = "Hello World"

        def f(a: amod.A, b: bmod.B, c: cmod.C): ...
        
        class C:
            def __init__(self: "C"):
                self.x = 0
                default = self.x or "10"
                self.x = default
        """
        )

        param_df = (
            pd.DataFrame(
                {
                    "file": [AnnotationTesting.FILENAME] * 4,
                    "category": [TypeCollectionCategory.CALLABLE_PARAMETER] * 4,
                    "qname": [f"f.{v}" for v in "abc"] + ["C.__init__.self"],
                    "anno": ["amod.A", "bmod.B", "cmod.C", "C"],
                }
            )
            .pipe(generate_qname_ssas_for_file)
            .pipe(pt.DataFrame[TypeCollectionSchema])
        )

        self.assertCodemod(
            AnnotationTesting.HINTLESS,
            after,
            tycol=param_df,
            context_override=codemod.CodemodContext(
                filename=AnnotationTesting.FILENAME,
                metadata_manager=metadata.FullRepoManager(
                    repo_root_dir=".", paths=[AnnotationTesting.FILENAME], providers=[]
                ),
            ),
        )

    def test_rettype(self):
        after = textwrap.dedent(
            f"""
        from __future__ import annotations
        import typing
        from typing import *

        a = 10
        a = "Hello World"

        def f(a, b, c) -> int: ...
        
        class C:
            def __init__(self) -> None:
                self.x = 0
                default = self.x or "10"
                self.x = default
        """
        )

        rettype_df = (
            pd.DataFrame(
                {
                    "file": [AnnotationTesting.FILENAME] * 2,
                    "category": [TypeCollectionCategory.CALLABLE_RETURN] * 2,
                    "qname": ["f", "C.__init__"],
                    "anno": ["int", "None"],
                }
            )
            .pipe(generate_qname_ssas_for_file)
            .pipe(pt.DataFrame[TypeCollectionSchema])
        )

        self.assertCodemod(
            AnnotationTesting.HINTLESS,
            after,
            tycol=rettype_df,
            context_override=codemod.CodemodContext(
                filename=AnnotationTesting.FILENAME,
                metadata_manager=metadata.FullRepoManager(
                    repo_root_dir=".", paths=[AnnotationTesting.FILENAME], providers=[]
                ),
            ),
        )

    def test_class_attribute(self):
        after = textwrap.dedent(
            f"""
        from __future__ import annotations
        import typing
        from typing import *

        a = 10
        a = "Hello World"

        def f(a, b, c): ...
        
        class C:
            a: int
            def __init__(self):
                self.x = 0
                default = self.x or "10"
                self.x = default
        """
        )

        attr_df = (
            pd.DataFrame(
                {
                    "file": [AnnotationTesting.FILENAME] * 1,
                    "category": [TypeCollectionCategory.CLASS_ATTR] * 1,
                    "qname": ["C.a"],
                    "anno": ["int"],
                }
            )
            .pipe(generate_qname_ssas_for_file)
            .pipe(pt.DataFrame[TypeCollectionSchema])
        )

        self.assertCodemod(
            AnnotationTesting.HINTLESS,
            after,
            tycol=attr_df,
            context_override=codemod.CodemodContext(
                filename=AnnotationTesting.FILENAME,
                metadata_manager=metadata.FullRepoManager(
                    repo_root_dir=".", paths=[AnnotationTesting.FILENAME], providers=[]
                ),
            ),
        )
