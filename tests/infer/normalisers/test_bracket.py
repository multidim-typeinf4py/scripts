from libcst import codemod


from scripts.infer.normalisers import bracket


class Test_RoundBrackets(codemod.CodemodTest):
    TRANSFORM = bracket.RoundBracketsToTuple

    def test_outer_tuple_rewritten(self):
        self.assertCodemod(
            before="a: (logger.Logger, int) = ...",
            after="a: Tuple[logger.Logger, int] = ...",
        )

    def test_inner_tuple_rewritten(self):
        self.assertCodemod(
            before="a: List[(str,)] = ...",
            after="a: List[Tuple[str]] = ...",
        )


class Test_SquareBrackets(codemod.CodemodTest):
    TRANSFORM = bracket.SquareBracketsToList

    def test_outer_list_rewritten(self):
        self.assertCodemod(
            before="a: [str] = ...",
            after="a: List[str] = ...",
        )

    def test_inner_list_ignored(self):
        self.assertCodemod(
            before="a: typing.Callable[[int], int] = ...",
            after="a: typing.Callable[[int], int] = ...",
        )


class Test_CurlyBraces(codemod.CodemodTest):
    TRANSFORM = bracket.CurlyBracesToDict

    def test_outer_dict_rewritten(self) -> None:
        self.assertCodemod(
            before=r"d: {}",
            after="d: Dict",
        )

    def test_inner_dict_rewritten(self) -> None:
        self.assertCodemod(
            before=r"d: dict[{}, str]",
            after="d: dict[Dict, str]",
        )
