import libcst
from libcst import codemod, metadata

from scripts.common.schemas import TypeCollectionCategory


class TaskPreprocessor(codemod.Codemod):
    def __init__(self, context: codemod.CodemodContext, task: TypeCollectionCategory) -> None:
        super().__init__(context)
        self.task = task

class AnnotationRemover(codemod.ContextAwareTransformer):
    METADATA_DEPENDENCIES = (
        metadata.ScopeProvider,
    )
    def __init__(self, context: codemod.CodemodContext, task: TypeCollectionCategory) -> None:
        super().__init__(context)
        self.task = task

    def leave_Param(self, original_node: libcst.Param, updated_node: libcst.Param) -> libcst.Param:
        if self.task is TypeCollectionCategory.CALLABLE_PARAMETER:
            return updated_node.with_changes(annotation=None)
        return updated_node

    def leave_FunctionDef(
        self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef
    ) -> libcst.FunctionDef:
        if self.task is TypeCollectionCategory.CALLABLE_RETURN:
            return updated_node.with_changes(returns=None)
        return updated_node

    def is_class_scope(self, original_node: libcst.CSTNode) -> bool:
        scope = self.get_metadata(metadata.ScopeProvider, original_node)
        is_class_scope = isinstance(scope, metadata.ClassScope)

        return is_class_scope