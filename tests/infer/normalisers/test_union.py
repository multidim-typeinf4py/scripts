from scripts.infer.normalisers import union

from libcst import codemod


class Test_Flatten(codemod.CodemodTest):
    TRANSFORM = union.Flatten

    def test_untouched(self) -> None:
        self.assertCodemod(
            before="a: Union[int, str] = ...",
            after="a: Union[int, str] = ...",
        )

    def test_simple_nestage(self) -> None:
        self.assertCodemod(
            before="a: Union[Union[int, str]] = ...",
            after="a: Union[int, str] = ...",
        )

    def test_flattenable_nestage(self) -> None:
        self.assertCodemod(
            before="a: Union[typing.Union[int, str], float] = ...",
            after="a: Union[int, str, float] = ...",
        )

    def test_advanced_nestage(self) -> None:
        self.assertCodemod(
            before="a: Union[Union[int, str], Union[float, bytes]] = ...",
            after="a: Union[int, str, float, bytes] = ...",
        )

    def test_deep_nestage(self) -> None:
        self.assertCodemod(
            before="a: Union[Union[Union[Union[int, str]]]] = ...",
            after="a: Union[int, str] = ...",
        )

    def test_empty_inner_union(self) -> None:
        self.assertCodemod(
            before="a: Union[Union, int] = ...",
            after="a: Union[int] = ..."
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
            after="a: typing.Union[int, str, bytes] = ...",
        )
