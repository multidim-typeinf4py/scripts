from libcst import codemod

from scripts.infer.normalisers import typing_aliases


class Test_LowercaseTypingAliases(codemod.CodemodTest):
    TRANSFORM = typing_aliases.LowercaseTypingAliases

    def test_list_plain(self) -> None:
        self.assertCodemod(
            before="a: List = ...",
            after="a: list = ...",
        )

    def test_list_subscript(self) -> None:
        self.assertCodemod(
            before="a: typing.List[int] = ...",
            after="a: list[int] = ...",
        )

    def test_list_nestage(self) -> None:
        self.assertCodemod(
            before="a: List[typing.List[int]] = ...",
            after="a: list[list[int]] = ...",
        )

    def test_tuple_plain(self) -> None:
        self.assertCodemod(
            before="a: Tuple = ...",
            after="a: tuple = ...",
        )

    def test_tuple_subscript(self) -> None:
        self.assertCodemod(
            before="a: typing.Tuple[int] = ...",
            after="a: tuple[int] = ...",
        )

    def test_tuple_nestage(self) -> None:
        self.assertCodemod(
            before="a: Tuple[typing.Tuple[int]] = ...",
            after="a: tuple[tuple[int]] = ...",
        )

    def test_dict_plain(self) -> None:
        self.assertCodemod(
            before="a: Dict = ...",
            after="a: dict = ...",
        )

    def test_dict_subscript(self) -> None:
        self.assertCodemod(
            before="a: typing.Dict[int, str] = ...",
            after="a: dict[int, str] = ...",
        )

    def test_dict_nestage(self) -> None:
        self.assertCodemod(
            before="a: Dict[typing.Dict[int, str], None] = ...",
            after="a: dict[dict[int, str], None] = ...",
        )
