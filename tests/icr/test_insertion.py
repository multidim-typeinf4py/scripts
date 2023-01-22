import textwrap


from libcst import codemod, metadata

from common._helper import generate_qname_ssas_for_file
from common.schemas import TypeCollectionCategory, TypeCollectionSchema

from icr.insertion import (
    FromQName2SSAQNameTransformer,
    FromSSAQName2QnameTransformer,
    TypeAnnotationApplierVisitor,
)

import pandas as pd
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


class Test_QName2QNameSSA(AnnotationTesting):
    TRANSFORM = FromQName2SSAQNameTransformer

    def test_qnames_transformed(self):
        self.assertCodemod(
            AnnotationTesting.HINTED,
            AnnotationTesting.HINTED_QNAME_SSA,
            annotations=AnnotationTesting.ANNOS,
        )


class Test_QNameSSA2QName(AnnotationTesting):
    TRANSFORM = FromSSAQName2QnameTransformer

    def test_qname_ssas_transformed(self):
        self.assertCodemod(
            AnnotationTesting.HINTED_QNAME_SSA,
            AnnotationTesting.HINTED,
            annotations=AnnotationTesting.ANNOS,
        )


class Test_CustomAnnotator(AnnotationTesting):
    TRANSFORM = TypeAnnotationApplierVisitor

    def test_correct_annotations_applied(self):
        filename = AnnotationTesting.ANNOS[TypeCollectionSchema.file].iloc[0]

        with_future_import = f"from __future__ import annotations\n{AnnotationTesting.HINTED}"

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
