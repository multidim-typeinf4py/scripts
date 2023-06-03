import textwrap


from libcst import codemod

from scripts.common import generate_qname_ssas_for_file
from scripts.common.schemas import TypeCollectionCategory

from scripts.infer.qname_transforms import (
    QName2SSATransformer,
    SSA2QNameTransformer,
)

import pandas as pd


class Test_QName2SSA(codemod.CodemodTest):
    TRANSFORM = QName2SSATransformer

    def test_qnames_transformed(self):
        self.assertCodemod(
            textwrap.dedent(
                """
                a = 10
                a = "Hello World"

                b, c = "Hello", 5

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

                bλ1, cλ1 = "Hello", 5

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
                }
            ).pipe(generate_qname_ssas_for_file),
        )

    def test_tuple_assignment_transformed(self):
        self.assertCodemod(
            "a, (b, c) = 1, 5, 20",
            "aλ1, (bλ1, cλ1) = 1, 5, 20",
            annotations=pd.DataFrame(
                {
                    "file": ["x.py"] * 3,
                    "category": [TypeCollectionCategory.VARIABLE] * 3,
                    "qname": ["a", "b", "c"],
                }
            ).pipe(generate_qname_ssas_for_file),
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
                }
            ).pipe(generate_qname_ssas_for_file),
        )

    def test_withitem_transformed(self):
        self.assertCodemod(
            """
            with open(file) as f:
                ...
            """,
            """
            with open(file) as fλ1:
                ...
            """,
            annotations=pd.DataFrame(
                {
                    "file": ["x.py"] * 1,
                    "category": [TypeCollectionCategory.VARIABLE] * 1,
                    "qname": ["f"],
                }
            ).pipe(generate_qname_ssas_for_file),
        )

    def test_hinting_transformed(self):
        self.assertCodemod(
            """
            a: int
            a, = 10, None
            """,
            """
            aλ1: int
            aλ1, = 10, None
            """,
            annotations=pd.DataFrame(
                {
                    "file": ["x.py"] * 1,
                    "category": [TypeCollectionCategory.VARIABLE] * 1,
                    "qname": ["a"],
                }
            ).pipe(generate_qname_ssas_for_file),
        )

    def test_branch_lowering(self):
        self.assertCodemod(
            """
            a: int | None
            if cond:
                a: int | None = λ__LOWERED_HINT_MARKER__λ; a = 5
            else:
                a: int | None = λ__LOWERED_HINT_MARKER__λ; a = None
            """,
            """
            aλ1: int | None
            if cond:
                aλ1: int | None = λ__LOWERED_HINT_MARKER__λ; aλ1 = 5
            else:
                aλ2: int | None = λ__LOWERED_HINT_MARKER__λ; aλ2 = None
            """,
            annotations=pd.DataFrame(
                {
                    "file": ["x.py"] * 2,
                    "category": [TypeCollectionCategory.VARIABLE] * 2,
                    "qname": ["a"] * 2,
                }
            ).pipe(generate_qname_ssas_for_file)
        )

    def test_class_attribute(self):
        self.assertCodemod(
            """
            class Clazz:
                a: int = 5
            """,
            """
            class Clazz:
                aλ1: int = 5
            """,
            annotations=pd.DataFrame(
                {
                    "file": ["x.py"],
                    "category": [TypeCollectionCategory.VARIABLE],
                    "qname": ["Clazz.a"],
                }
            ).pipe(generate_qname_ssas_for_file)
        )

    def test_instance_attribute(self):
        self.assertCodemod(
            """
            class Clazz:
                a: int
            """,
            """
            class Clazz:
                aλ1: int
            """,
            annotations=pd.DataFrame(
                {
                    "file": ["x.py"],
                    "category": [TypeCollectionCategory.VARIABLE],
                    "qname": ["Clazz.a"],
                }
            ).pipe(generate_qname_ssas_for_file)
        )


class Test_SSA2QName(codemod.CodemodTest):
    TRANSFORM = SSA2QNameTransformer

    def test_qnames_transformed(self):
        self.assertCodemod(
            textwrap.dedent(
                """
                aλ1 = 10
                aλ2 = "Hello World"

                bλ1, cλ1 = "Hello", 5

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

                b, c = "Hello", 5

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
                }
            ).pipe(generate_qname_ssas_for_file),
        )

    def test_tuple_assignment_transformed(self):
        self.assertCodemod(
            "aλ1, (bλ1, cλ1) = 1, 5, 20",
            "a, (b, c) = 1, 5, 20",
            annotations=pd.DataFrame(
                {
                    "file": ["x.py"] * 3,
                    "category": [TypeCollectionCategory.VARIABLE] * 3,
                    "qname": ["a", "b", "c"],
                }
            ).pipe(generate_qname_ssas_for_file),
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
                }
            ).pipe(generate_qname_ssas_for_file),
        )

    def test_hinting_transformed(self):
        self.assertCodemod(
            """
            aλ1: int
            aλ1, = 10, None
            """,
            """
            a: int
            a, = 10, None
            """,
            annotations=pd.DataFrame(
                {
                    "file": ["x.py"] * 1,
                    "category": [TypeCollectionCategory.VARIABLE] * 1,
                    "qname": ["a"],
                }
            ).pipe(generate_qname_ssas_for_file),
        )

    def test_branch_lowering(self):
        self.assertCodemod(
            """
            if cond:
                aλ1: int | None = λ__LOWERED_HINT_MARKER__λ; aλ1 = 5
            else:
                aλ2: int | None = λ__LOWERED_HINT_MARKER__λ; aλ2 = None
            """,
            """
            if cond:
                a: int | None = λ__LOWERED_HINT_MARKER__λ; a = 5
            else:
                a: int | None = λ__LOWERED_HINT_MARKER__λ; a = None
            """,
            annotations=pd.DataFrame(
                {
                    "file": ["x.py"] * 2,
                    "category": [TypeCollectionCategory.VARIABLE] * 2,
                    "qname": ["a"] * 2,
                }
            ).pipe(generate_qname_ssas_for_file)
        )

    def test_class_attribute(self):
        self.assertCodemod(
            """
            class Clazz:
                aλ1: int = 5
            """,
            """
            class Clazz:
                a: int = 5
            """,
            annotations=pd.DataFrame(
                {
                    "file": ["x.py"],
                    "category": [TypeCollectionCategory.VARIABLE],
                    "qname": ["Clazz.a"],
                }
            ).pipe(generate_qname_ssas_for_file)
        )

    def test_instance_attribute(self):
        self.assertCodemod(
            """
            class Clazz:
                aλ1: int
            """,
            """
            class Clazz:
                a: int
            """,
            annotations=pd.DataFrame(
                {
                    "file": ["x.py"],
                    "category": [TypeCollectionCategory.VARIABLE],
                    "qname": ["Clazz.a"],
                }
            ).pipe(generate_qname_ssas_for_file)
        )