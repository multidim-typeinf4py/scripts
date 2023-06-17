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

    def test_set_plain(self) -> None:
        self.assertCodemod(
            before="a: Set = ...",
            after="a: set = ...",
        )

    def test_set_subscript(self) -> None:
        self.assertCodemod(
            before="a: typing.Set[int, str] = ...",
            after="a: set[int, str] = ...",
        )

    def test_set_nestage(self) -> None:
        self.assertCodemod(
            before="a: Set[typing.Set[int, str], None] = ...",
            after="a: set[set[int, str], None] = ...",
        )


class Test_TextToStr(codemod.CodemodTest):
    TRANSFORM = typing_aliases.TextToStr

    def test_outer(self) -> None:
        self.assertCodemod(
            before="a: typing.Text = ...",
            after="a: str = ...",
        )

        self.assertCodemod(
            before="a: Text = ...",
            after="a: str = ...",
        )

    def test_inner(self) -> None:
        self.assertCodemod(
            before="a: dict[typing.Text, typing.Text] = ...",
            after="a: dict[str, str] = ...",
        )

        self.assertCodemod(
            before="a: dict[Text, Text] = ...",
            after="a: dict[str, str] = ...",
        )


class Test_RemoveOuterOptional(codemod.CodemodTest):
    TRANSFORM = typing_aliases.RemoveOuterOptional

    def test_outer(self) -> None:
        self.assertCodemod(
            before="a: Optional[int] = ...",
            after="a: int = ...",
        )

        self.assertCodemod(
            before="a: typing.Optional[int] = ...",
            after="a: int = ...",
        )

    def test_inner(self) -> None:
        self.assertCodemod(
            before="a: dict[Optional[int], str] = ...",
            after="a: dict[Optional[int], str] = ...",
        )

        self.assertCodemod(
            before="a: dict[typing.Optional[int], str] = ...",
            after="a: dict[typing.Optional[int], str] = ...",
        )


class Test_RemoveOuterFinal(codemod.CodemodTest):
    TRANSFORM = typing_aliases.RemoveOuterFinal

    def test_outer(self) -> None:
        self.assertCodemod(
            before="a: Final[int] = ...",
            after="a: int = ...",
        )

        self.assertCodemod(
            before="a: typing.Final[int] = ...",
            after="a: int = ...",
        )

    def test_inner(self) -> None:
        self.assertCodemod(
            before="a: dict[Final[int], str] = ...",
            after="a: dict[Final[int], str] = ...",
        )

        self.assertCodemod(
            before="a: dict[typing.Final[int], str] = ...",
            after="a: dict[typing.Final[int], str] = ...",
        )