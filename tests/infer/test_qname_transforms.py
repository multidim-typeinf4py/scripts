import textwrap


from libcst import codemod, metadata

from common.ast_helper import generate_qname_ssas_for_file
from common.schemas import TypeCollectionCategory, TypeCollectionSchema

from infer.insertion import (
    QName2SSATransformer,
    SSA2QNameTransformer,
    TypeAnnotationApplierTransformer,
)

import pandas as pd
from pandas._libs import missing
import pandera.typing as pt


class Test_QName2SSA(codemod.CodemodTest):
    TRANSFORM = QName2SSATransformer

    def test_qnames_transformed(self):
        self.assertCodemod(
            textwrap.dedent(
                """
                a = 10
                a = "Hello World"

                (b, c) = "Hello", 5

                def f(a, b, c): ...
                
                class C:
                    def __init__(self):
                        self.x = 0
                        default = self.x or "10"
                        self.x = default
                """
            ),
            textwrap.dedent(
                """
                aλ1 = 10
                aλ2 = "Hello World"

                (bλ1, cλ1) = "Hello", 5

                def f(a, b, c): ...
                
                class C:
                    def __init__(self):
                        self.xλ1 = 0
                        defaultλ1 = self.x or "10"
                        self.xλ2 = default
                """
            ),
            annotations=pd.DataFrame(
                {
                    "file": ["x.py"] * 7,
                    "category": [TypeCollectionCategory.VARIABLE] * 7,
                    "qname": ["a"] * 2
                    + ["b", "c"]
                    + [f"C.__init__.{v}" for v in ("self.x", "default", "self.x")],
                    "anno": ["int", "str", "str", "int", "int", "str", "str"],
                }
            )
            .pipe(generate_qname_ssas_for_file)
            .pipe(pt.DataFrame[TypeCollectionSchema]),
        )

    def test_tuple_assignment_transformed(self):
        self.assertCodemod(
            "a, (b, c) = 1, 5, 20",
            "(aλ1, (bλ1, cλ1)) = 1, 5, 20",
            annotations=pd.DataFrame(
                {
                    "file": ["x.py"] * 3,
                    "category": [TypeCollectionCategory.VARIABLE] * 3,
                    "qname": ["a", "b", "c"],
                    "anno": ["int", "int", "int"],
                }
            )
            .pipe(generate_qname_ssas_for_file)
            .pipe(pt.DataFrame[TypeCollectionSchema]),
        )

    def test_list_assignment_transformed(self):
        self.assertCodemod(
            "[a, [b, *c]] = 1, 5, 20",
            "[aλ1, [bλ1, *cλ1]] = 1, 5, 20",
            annotations=pd.DataFrame(
                {
                    "file": ["x.py"] * 3,
                    "category": [TypeCollectionCategory.VARIABLE] * 3,
                    "qname": ["a", "b", "c"],
                    "anno": ["int", "int", "int"],
                }
            )
            .pipe(generate_qname_ssas_for_file)
            .pipe(pt.DataFrame[TypeCollectionSchema]),
        )


class Test_SSA2QName(codemod.CodemodTest):
    TRANSFORM = SSA2QNameTransformer

    def test_qnames_transformed(self):
        self.assertCodemod(
            textwrap.dedent(
                """
                aλ1 = 10
                aλ2 = "Hello World"

                (bλ1, cλ1) = "Hello", 5

                def f(a, b, c): ...
                
                class C:
                    def __init__(self):
                        self.xλ1 = 0
                        defaultλ1 = self.x or "10"
                        self.xλ2 = default
                """
            ),
            textwrap.dedent(
                """
                a = 10
                a = "Hello World"

                (b, c) = "Hello", 5

                def f(a, b, c): ...
                
                class C:
                    def __init__(self):
                        self.x = 0
                        default = self.x or "10"
                        self.x = default
                """
            ),
            annotations=pd.DataFrame(
                {
                    "file": ["x.py"] * 7,
                    "category": [TypeCollectionCategory.VARIABLE] * 7,
                    "qname": ["a"] * 2
                    + ["b", "c"]
                    + [f"C.__init__.{v}" for v in ("self.x", "default", "self.x")],
                    "anno": ["int", "str", "str", "int", "int", "str", "str"],
                }
            )
            .pipe(generate_qname_ssas_for_file)
            .pipe(pt.DataFrame[TypeCollectionSchema]),
        )

    def test_tuple_assignment_transformed(self):
        self.assertCodemod(
            "(aλ1, (bλ1, cλ1)) = 1, 5, 20",
            "(a, (b, c)) = 1, 5, 20",
            annotations=pd.DataFrame(
                {
                    "file": ["x.py"] * 3,
                    "category": [TypeCollectionCategory.VARIABLE] * 3,
                    "qname": ["a", "b", "c"],
                    "anno": ["int", "int", "int"],
                }
            )
            .pipe(generate_qname_ssas_for_file)
            .pipe(pt.DataFrame[TypeCollectionSchema]),
        )

    def test_list_assignment_transformed(self):
        self.assertCodemod(
            "[aλ1, [bλ1, *cλ1]] = 1, 5, 20",
            "[a, [b, *c]] = 1, 5, 20",
            annotations=pd.DataFrame(
                {
                    "file": ["x.py"] * 3,
                    "category": [TypeCollectionCategory.VARIABLE] * 3,
                    "qname": ["a", "b", "c"],
                    "anno": ["int", "int", "int"],
                }
            )
            .pipe(generate_qname_ssas_for_file)
            .pipe(pt.DataFrame[TypeCollectionSchema]),
        )