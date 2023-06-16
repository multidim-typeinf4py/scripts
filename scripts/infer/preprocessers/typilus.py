import libcst
from typet5.experiments import typilus, utils

from .base import TaskPreprocessor
from .tt5 import _ConditionalTT5AnnoRemover


class TypilusPreprocessor(TaskPreprocessor):
    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        syntax_removed = utils.remove_newer_syntax(tree, typilus.TypilusSupportedSyntax)

        annotations_removed = _ConditionalTT5AnnoRemover(
            context=self.context, task=self.task
        ).transform_module(syntax_removed)
        return annotations_removed
