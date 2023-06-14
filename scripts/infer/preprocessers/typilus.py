import libcst
from libcst import codemod

from .base import TaskPreprocessor
from scripts.common.schemas import TypeCollectionCategory

from typet5.experiments import typilus
from typet5.experiments import utils


class TypilusPreprocessor(TaskPreprocessor):
    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        syntax_removed = utils.remove_newer_syntax(tree, typilus.TypilusSupportedSyntax)
        return TypilusAnnoRemover(
            context=self.context,
            task=self.task,
        ).transform_module(syntax_removed)


class TypilusAnnoRemover(codemod.ContextAwareTransformer):
    def __init__(self, context: codemod.CodemodContext, task: TypeCollectionCategory) -> None:
        super().__init__(context)
        self.task = task
