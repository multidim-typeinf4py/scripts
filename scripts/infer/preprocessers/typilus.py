import libcst
from libcst import codemod
from typet5.experiments import typilus, utils

from .base import TaskPreprocessor
from .tt5 import TT5AnnotationRemover


class TypilusPreprocessor(TaskPreprocessor):
    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        syntax_removed = utils.remove_newer_syntax(tree, typilus.TypilusSupportedSyntax)
        without_fstrings = FStringRewriter(
            context=self.context
        ).transform_module(syntax_removed)

        annotations_removed = TT5AnnotationRemover(
            context=self.context, task=self.task
        ).transform_module(without_fstrings)
        return annotations_removed


class FStringRewriter(codemod.ContextAwareTransformer):
    def leave_FormattedString(
        self, original_node: libcst.FormattedString, updated_node: libcst.FormattedString
    ) -> libcst.BaseExpression:
        # typed_ast struggles to handle f-strings correctly
        # e.g.: f"{repo or ''}" fails due to the usage of boolean operators
        # however, f-strings are all strings, therefore simply return some form of SimpleString
        return libcst.SimpleString("'<REWRITTEN-FSTRING>'")
