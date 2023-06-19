from libcst import codemod

from scripts.infer.normalisers import literal_to_base

class Test_LiteralToBaseClass(codemod.CodemodTest):
    TRANSFORM = literal_to_base.LiteralToBaseClass

    def test_builtins_false_to_bool(self):
        self.assertCodemod(
            before="a: builtins.False",
            after="a: bool",
        )

    def test_builtins_true_to_bool(self):
        self.assertCodemod(
            before="a: builtins.True",
            after="a: bool",
        )