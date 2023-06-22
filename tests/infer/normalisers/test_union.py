from scripts.infer.normalisers import union

from libcst import codemod


class Test_UnionNormaliser(codemod.CodemodTest):
    TRANSFORM = union.UnionNormaliser

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

    def test_optional_to_union(self) -> None:
        self.assertCodemod(
            before="a: Optional[int]",
            after="a: typing.Union[None, int]"
        )

        self.assertCodemod(
            before="a: typing.Optional[int]",
            after="a: typing.Union[None, int]"
        )

    def test_do_not_sort_other_generics(self) -> None:
        self.assertCodemod(
            before="a: dict[str, int]",
            after="a: dict[str, int]",
        )

    def test_more_sorting(self) -> None:
        self.assertCodemod(
            before="data: typing.Union[builtins.str, builtins.bytes]",
            after="data: typing.Union[builtins.bytes, builtins.str]",
        )
        self.assertCodemod(
            before="data: typing.Union[builtins.bytes, builtins.str]",
            after="data: typing.Union[builtins.bytes, builtins.str]",
        )
    def test_nested_optional(self) -> None:
        self.assertCodemod(
            before="a: typing.Union[typing.Optional[builtins.str]]",
            after="a: typing.Union[None, builtins.str]"
        )