import libcst
from libcst import codemod
from scripts.infer.preprocessers import static

from scripts.common.schemas import TypeCollectionCategory
from .samples import code

class Test_StaticPreprocessing(codemod.CodemodTest):
    TRANSFORM = static.StaticPreprocessor

    def test_variable_removal(self) -> None:
        self.assertCodemod(
            before=code,
            after="""
            def f(a: int, b: int, c: int) -> int:
                v = a + b + c
                return v

            class Interface:
                attr = int()
            """,
            task=TypeCollectionCategory.VARIABLE,
        )

    def test_parameter_removal(self) -> None:
        self.assertCodemod(
            before=code,
            after="""
            def f(a, b, c) -> int:
                v: int
                v: int = a + b + c
                return v

            class Interface:
                attr: int
            """,
            task=TypeCollectionCategory.CALLABLE_PARAMETER,
        )

    def test_return_removal(self) -> None:
        self.assertCodemod(
            before=code,
            after="""
            def f(a: int, b: int, c: int):
                v: int
                v: int = a + b + c
                return v

            class Interface:
                attr: int
            """,
            task=TypeCollectionCategory.CALLABLE_RETURN,
        )