import textwrap
import libcst
from libcst import codemod
from scripts.infer.preprocessers import typilus

from scripts.common.schemas import TypeCollectionCategory
from .samples import code


class Test_Typilus(codemod.CodemodTest):
    TRANSFORM = typilus.TypilusPreprocessor

    def assertWithPreamble(self, before: str, after: str, task: TypeCollectionCategory) -> None:
        before = textwrap.dedent(before)
        after = f"from typing import Any, List, Tuple, Dict, Set, Union, Type, Callable # SPOT{textwrap.dedent(after)}"

        self.assertCodemod(before, after, task=task)

    def test_variable_removal(self) -> None:
        self.assertWithPreamble(
            before=code,
            after="""
            def f(a: int, b: int, c: int) -> int:
                v: ...
                v = a + b + c
                return v

            class Interface:
                attr: ...
            """,
            task=TypeCollectionCategory.VARIABLE,
        )

    def test_parameter_removal(self) -> None:
        self.assertWithPreamble(
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
        self.assertWithPreamble(
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
