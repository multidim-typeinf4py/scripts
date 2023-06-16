import libcst
from libcst import codemod, metadata

from .base import TaskPreprocessor, AnnotationRemover
from .tt5 import _ConditionalTT5AnnoRemover
from scripts.common.schemas import TypeCollectionCategory

from typet5.experiments import typilus, utils


class TypilusPreprocessor(TaskPreprocessor):
    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        syntax_removed = utils.remove_newer_syntax(tree, typilus.TypilusSupportedSyntax)

        annotations_removed = _ConditionalTT5AnnoRemover(
            context=self.context, task=self.task
        ).transform_module(syntax_removed)
        return annotations_removed
