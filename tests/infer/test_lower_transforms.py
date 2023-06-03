import textwrap
from libcst import codemod as c
import libcst

from scripts.infer.lower_transforms import (
    LoweringTransformer,
    UnloweringTransformer,
)


class Test_Lowering(c.CodemodTest):
    TRANSFORM = LoweringTransformer

    def test_nonlowered_branchless(self):
        code = textwrap.dedent(
            """
            a: int = 10
            a = 20
            a = 50
            """
        )
        self.assertCodemod(before=code, after=code)

    def test_hint_lowered_into_branch(self):
        self.assertCodemod(
            before="""
            a: int | None
            if cond:
                a = 10
            else:
                a = None
            """,
            after="""
            a: int | None
            if cond:
                a: int | None = λ__LOWERED_HINT_MARKER__λ; a = 10
            else:
                a: int | None = λ__LOWERED_HINT_MARKER__λ; a = None
            """,
        )

    def test_annassign_lowered_into_branch(self):
        self.assertCodemod(
            before="""
            a: int | None = None
            if cond:
                a = 10
            """,
            after="""
            a: int | None = None
            if cond:
                a: int | None = λ__LOWERED_HINT_MARKER__λ; a = 10
            """,
        )

    def test_no_unneeded_lower(self):
        self.assertCodemod(
            before="""
            a: int | None = None
            if cond:
                a: int = 10
                a = 20

            a = 30
            """,
            after="""
            a: int | None = None
            if cond:
                a: int = 10
                a = 20

            a = 30
            """,
        )


class Test_Unlowering(c.CodemodTest):
    TRANSFORM = UnloweringTransformer

    def assertWithLower(self, code: str) -> None:
        code_module = libcst.parse_module(textwrap.dedent(code))

        context = c.CodemodContext()
        lowered = LoweringTransformer(context).transform_module(code_module)
        unlowered = UnloweringTransformer(context).transform_module(lowered)

        print("before:", code_module.code)
        print("lowered:", lowered.code)
        print("unlowered:", unlowered.code)

        self.assertCodeEqual(code_module.code, unlowered.code)

    def test_nonlowered_branchless(self):
        self.assertWithLower(
            code="""
            a: int = 10
            a = 20
            a = 50
            """
        )

    def test_hint_lowered_into_branch(self):
        self.assertWithLower(
            code="""
            a: int | None
            if cond:
                a = 10
            else:
                a = None
            """
        )

    def test_annassign_lowered_into_branch(self):
        self.assertWithLower(
            code="""
            a: int | None = None
            if cond:
                a = 10
            """,
        )

    def test_no_unneeded_lower(self):
        self.assertWithLower(
            code="""
            a: int | None = None
            if cond:
                a: int = 10
                a = 20

            a = 30
            """
        )
