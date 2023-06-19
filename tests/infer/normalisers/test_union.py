from scripts.infer.normalisers import union

from libcst import codemod


class Test_Flatten(codemod.CodemodTest):
    TRANSFORM = union.FlattenAndSort

    def test_untouched(self) -> None:
        self.assertCodemod(
            before="a: typing.Union[int, str] = ...",
            after="a: typing.Union[int, str] = ...",
        )

    def test_simple_nestage(self) -> None:
        self.assertCodemod(
            before="a: typing.Union[typing.Union[str, int]] = ...",
            after="a: typing.Union[int, str] = ...",
        )

    def test_flattenable_nestage(self) -> None:
        self.assertCodemod(
            before="a: typing.Union[typing.Union[int, str], float] = ...",
            after="a: typing.Union[float, int, str] = ...",
        )

    def test_advanced_nestage(self) -> None:
        self.assertCodemod(
            before="a: typing.Union[typing.Union[int, str], typing.Union[float, bytes]] = ...",
            after="a: typing.Union[bytes, float, int, str] = ...",
        )

    def test_deep_nestage(self) -> None:
        self.assertCodemod(
            before="a: typing.Union[typing.Union[typing.Union[typing.Union[int, str]]]] = ...",
            after="a: typing.Union[int, str] = ...",
        )

    def test_empty_inner_union(self) -> None:
        self.assertCodemod(
            before="a: typing.Union[typing.Union, int] = ...",
            after="a: typing.Union[int] = ..."
        )


class Test_Pep604(codemod.CodemodTest):
    TRANSFORM = union.Pep604

    def test_simple_unionage(self) -> None:
        self.assertCodemod(
            before="a: int | str = ...",
            after="a: typing.Union[int, str] = ...",
        )

    def test_advanced_unionage(self) -> None:
        self.assertCodemod(
            before="a: int | str | bytes = ...",
            after="a: typing.Union[bytes, int, str] = ...",
        )
