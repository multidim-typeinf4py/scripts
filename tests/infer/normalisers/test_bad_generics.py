import libcst
from libcst import codemod


from scripts.infer.normalisers import bad_generics



class Test_BadGenerics(codemod.CodemodTest):
    TRANSFORM = bad_generics.BadGenericsNormaliser

    def test_outer_list_rewritten(self):
        self.assertCodemod(
            before="a: [str] = ...",
            after="a: typing.List[str] = ...",
        )

    def test_multi_list_to_list_union(self):
        self.assertCodemod(
            before="a: [str, int]",
            after="a: typing.List[typing.Union[str, int]]"
        )

    def test_inner_list_ignored(self):
        self.assertCodemod(
            before="a: typing.Callable[[int], int] = ...",
            after="a: typing.Callable[[int], int] = ...",
        )

    def test_outer_tuple_rewritten(self):
        self.assertCodemod(
            before="a: (logger.Logger, int) = ...",
            after="a: typing.Tuple[logger.Logger, int] = ...",
        )

    def test_inner_tuple_rewritten(self):
        self.assertCodemod(
            before="a: List[(str,)] = ...",
            after="a: List[typing.Tuple[str]] = ...",
        )

    def test_outer_dict_rewritten(self) -> None:
        self.assertCodemod(
            before=r"d: {}",
            after="d: typing.Dict",
        )

    def test_inner_dict_rewritten(self) -> None:
        self.assertCodemod(
            before=r"d: dict[{}, str]",
            after="d: dict[typing.Dict, str]",
        )


    #def test_combined_rewritten(self) -> None:
    #    self.assertCodemod(
    #        before="[{}]",
    #        after="typing.List[typing.Dict]"
    #    )

    def test_builtins_false_to_bool(self):
        illegal_syntax = libcst.AnnAssign(target=libcst.Name("a"), annotation=libcst.Annotation(
            libcst.Attribute(libcst.Name("builtins"), libcst.Name("False"))
        ))
        as_module = libcst.Module([libcst.SimpleStatementLine([illegal_syntax])])
        transformed = bad_generics.BadGenericsNormaliser(context=codemod.CodemodContext()).transform_module(as_module)
        self.assertCodeEqual(
            expected="a: builtins.bool",
            actual=transformed.code
        )


    def test_builtins_true_to_bool(self):
        illegal_syntax = libcst.AnnAssign(target=libcst.Name("a"), annotation=libcst.Annotation(
            libcst.Attribute(libcst.Name("builtins"), libcst.Name("True"))
        ))
        as_module = libcst.Module([libcst.SimpleStatementLine([illegal_syntax])])
        transformed = bad_generics.BadGenericsNormaliser(context=codemod.CodemodContext()).transform_module(as_module)
        self.assertCodeEqual(
            expected="a: builtins.bool",
            actual=transformed.code
        )

    def test_boolean_literal_in_subscript(self):
        illegal_syntax = libcst.AnnAssign(target=libcst.Name("a"), annotation=libcst.Annotation(
            libcst.Subscript(libcst.Name("dict"), slice=[
                libcst.SubscriptElement(libcst.Index(libcst.Attribute(libcst.Name("builtins"), libcst.Name("False"))))
            ])
        ))
        as_module = libcst.Module([libcst.SimpleStatementLine([illegal_syntax])])
        transformed = bad_generics.BadGenericsNormaliser(context=codemod.CodemodContext()).transform_module(as_module)
        self.assertCodeEqual(
            expected="a: dict[builtins.bool]",
            actual=transformed.code
        )

    def test_literal_conversion(self):
        self.assertCodemod(
            before="a: typing_extensions.Literal[b'Report-To']",
            after="a: builtins.bytes",
        )

        self.assertCodemod(
            before="a: typing_extensions.Literal['Report-To']",
            after="a: builtins.str",
        )

        self.assertCodemod(
            before="a: typing_extensions.Literal[42]",
            after="a: builtins.int",
        )

        self.assertCodemod(
            before="a: typing_extensions.Literal[42.5]",
            after="a: builtins.float",
        )

        self.assertCodemod(
            before='type: Literal["default"] = "default"',
            after='type: builtins.str = "default"',
        )

        self.assertCodemod(
            before='type: Literal[None]',
            after='type: None',
        )