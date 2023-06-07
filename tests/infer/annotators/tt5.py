import textwrap

import libcst
from libcst import codemod
from typet5.static_analysis import SignatureMap, ProjectPath, VariableSignature

from scripts.infer.annotators.tt5 import TT5FileApplier


class Test_TT5(codemod.CodemodTest):
    TRANSFORM = TT5FileApplier

    def assertTypeT5Codemod(
        self, before: str, after: str, sigmap: SignatureMap
    ) -> None:
        after = f"from typing import Any, List, Tuple, Dict, Set, Union, Type, Callable # SPOT\n{textwrap.dedent(after)}"
        self.assertCodemod(
            before,
            after,
            context_override=codemod.CodemodContext(
                filename="x.py", full_module_name="x"
            ),
            sigmap=sigmap,
        )

    def test_outer_tuple_rewritten(self):
        self.assertTypeT5Codemod(
            before="a = ...",
            after="a: Tuple[logger.Logger, int] = ...",
            sigmap={
                ProjectPath(module="x", path="a"): VariableSignature(
                    annot=libcst.Annotation(
                        libcst.parse_expression("(logger.Logger, int)")
                    ),
                    in_class=False,
                )
            },
        )

    def test_inner_tuple_rewritten(self):
        self.assertTypeT5Codemod(
            before="a = ...",
            after="a: List[Tuple[str]] = ...",
            sigmap={
                ProjectPath(module="x", path="a"): VariableSignature(
                    annot=libcst.Annotation(
                        libcst.parse_expression("List[(str,)]")
                    ),
                    in_class=False,
                )
            },
        )


    def test_outer_list_rewritten(self):
        self.assertTypeT5Codemod(
            before="a = ...",
            after="a: List[str] = ...",
            sigmap={
                ProjectPath(module="x", path="a"): VariableSignature(
                    annot=libcst.Annotation(
                        libcst.parse_expression("[str]")
                    ),
                    in_class=False,
                )
            },
        )


    def test_inner_list_ignored(self):
        self.assertTypeT5Codemod(
            before="a = ...",
            after="a: typing.Callable[[int], int] = ...",
            sigmap={
                ProjectPath(module="x", path="a"): VariableSignature(
                    annot=libcst.Annotation(
                        libcst.parse_expression("typing.Callable[[int], int]")
                    ),
                    in_class=False,
                )
            },
        )
