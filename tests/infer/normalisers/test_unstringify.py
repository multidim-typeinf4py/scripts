from libcst import codemod

from scripts.infer.normalisers import unstringify

class Test_Unstringify(codemod.CodemodTest):
    TRANSFORM = unstringify.Unquote

    def test_outer_removed(self):
        self.assertCodemod(
            before="a: 'int'",
            after="a: int"
        )

    def test_inner_removed(self):
        self.assertCodemod(
            before="a: typing.Type['AbstractExtractors']",
            after="a: typing.Type[AbstractExtractors]",
        )

    def test_annotated_metadata(self):
        self.assertCodemod(
            before="a: Annotated[T, 'x']",
            after="a: T"
        )