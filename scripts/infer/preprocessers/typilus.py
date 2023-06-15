import libcst
from libcst import codemod, metadata

from .base import TaskPreprocessor, AnnotationRemover
from scripts.common.schemas import TypeCollectionCategory

from typet5.experiments import typilus
from typet5.experiments import utils


class TypilusPreprocessor(TaskPreprocessor):
    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        syntax_removed = utils.remove_newer_syntax(tree, typilus.TypilusSupportedSyntax)

        annotations_removed = _TypilusAnnotationRemover(
            context=self.context, task=self.task
        ).transform_module(syntax_removed)
        return annotations_removed


class _TypilusAnnotationRemover(AnnotationRemover):
    METADATA_DEPENDENCIES = (metadata.ScopeProvider,)

    def leave_AnnAssign(
        self, original_node: libcst.AnnAssign, updated_node: libcst.AnnAssign
    ) -> libcst.Assign | libcst.AnnAssign | libcst.RemovalSentinel:
        if self.task is not TypeCollectionCategory.VARIABLE:
            return updated_node

        # Remove as much as we can without losing datapoints, 
        # typilus does not seem to care for annotations
    
        is_class_scope = self.is_class_scope(original_node.target)

        # a: int inside of class
        if original_node.value is None and is_class_scope:
            # a: int -> a = ...
            return libcst.Assign(
                targets=[libcst.AssignTarget(original_node.target)],
                value=libcst.parse_expression("..."),
            )

        # a: int outside of class
        elif original_node.value is None and not is_class_scope:
            # a: int -> removed
            return libcst.RemoveFromParent()

        else:
            # a: int = 5 -> a = 5
            # both inside of and outside of classes
            return updated_node.with_changes(annotation=None)
    
            


