import libcst
from libcst import codemod, metadata

from scripts.infer.preprocessers.base import AnnotationRemover, TaskPreprocessor


class MonkeyPreprocessor(TaskPreprocessor):
    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        rewritten = MonkeyAnnotationRemover(context=self.context).transform_module(tree)
        return rewritten


class MonkeyAnnotationRemover(codemod.ContextAwareTransformer):
    METADATA_DEPENDENCIES = (metadata.ScopeProvider,)

    def leave_Param(
        self, original_node: libcst.Param, updated_node: libcst.Param
    ) -> libcst.Param:
        return updated_node.with_changes(annotation=None)

    def leave_FunctionDef(
        self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef
    ) -> libcst.FunctionDef:
        return updated_node.with_changes(returns=None)

    def leave_AnnAssign(
        self, original_node: libcst.AnnAssign, updated_node: libcst.AnnAssign
    ) -> libcst.Assign | libcst.AnnAssign | libcst.RemovalSentinel:
        is_class_scope = self.is_class_scope(original_node.target)
        if is_class_scope and original_node.value is None:
            # a: int -> a = int()
            return libcst.Assign(
                targets=[libcst.AssignTarget(updated_node.target)],
                value=libcst.Call(updated_node.annotation.annotation),
            )

        elif not is_class_scope and original_node.value is None:
            # a: int -> removed
            return libcst.RemoveFromParent()

        else:
            # a: int = 5 -> a = 5
            # both inside of and outside of classes
            return libcst.Assign(
                targets=[libcst.AssignTarget(updated_node.target)],
                value=updated_node.value,
            )

    def is_class_scope(self, original_node: libcst.CSTNode) -> bool:
        scope = self.get_metadata(metadata.ScopeProvider, original_node)
        is_class_scope = isinstance(scope, metadata.ClassScope)

        return is_class_scope
