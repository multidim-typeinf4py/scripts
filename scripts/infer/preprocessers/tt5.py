import libcst
from libcst import codemod

from scripts.infer.inference._base import TypeCollectionCategory
from scripts.infer.preprocessers.base import AnnotationRemover, TaskPreprocessor
from typet5.static_analysis import mask_assign_type


class TT5Preprocessor(TaskPreprocessor):
    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        rewritten = TT5AnnotationRemover(context=self.context, task=self.task).transform_module(tree)
        return rewritten


class TT5AnnotationRemover(AnnotationRemover):
    # Adapted from TT5's codebase
    """Removes all type annotations when possible or replace them with a special symbol."""

    def __init__(
        self, context: codemod.CodemodContext, task: TypeCollectionCategory, type_mask="..."
    ) -> None:
        super().__init__(context, task)
        self.type_mask = libcst.Ellipsis() if type_mask == "..." else libcst.Name(type_mask)

    def leave_FunctionDef(
        self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef
    ) -> libcst.FunctionDef:
        if self.task is not TypeCollectionCategory.CALLABLE_RETURN:
            return updated_node
        return updated_node.with_changes(returns=None)

    def leave_Param(self, original_node: libcst.Param, updated_node: libcst.Param) -> libcst.Param:
        if self.task is not TypeCollectionCategory.CALLABLE_PARAMETER:
            return updated_node
        return updated_node.with_changes(annotation=None)

    def leave_AnnAssign(self, original_node: libcst.AnnAssign, updated_node: libcst.AnnAssign):
        if self.task is not TypeCollectionCategory.VARIABLE:
            return updated_node
        updated_node = updated_node.with_changes(annotation=libcst.Annotation(self.type_mask))
        return mask_assign_type(updated_node)
