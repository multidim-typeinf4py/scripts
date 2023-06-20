from libcst import codemod

from scripts.infer.normalisers.remove_anys import RemoveAnys


class Test_RemoveAnys(codemod.CodemodTest):
    TRANSFORM = RemoveAnys

    def test_untouched(self):
        self.assertCodemod(
            before="a: List[int]",
            after="a: List[int]",
        )

    def test_single_any_removed(self):
        self.assertCodemod(
            before="a: List[typing.Any]",
            after="a: List",
        )

    def test_multiple_any_removed(self):
        self.assertCodemod(
            before="a: Union[typing.Any, typing.Any]",
            after="a: Union",
        )
