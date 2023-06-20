import libcst
from typet5.experiments import typilus, utils

from .base import TaskPreprocessor
from .tt5 import TT5AnnotationRemover


class TypilusPreprocessor(TaskPreprocessor):
    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        syntax_removed = utils.remove_newer_syntax(tree, typilus.TypilusSupportedSyntax)

        annotations_removed = TT5AnnotationRemover(
            context=self.context, task=self.task
        ).transform_module(syntax_removed)
        return annotations_removed
