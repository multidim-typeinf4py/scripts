import libcst
from libcst import codemod

class MonkeyPreprocessor(codemod.ContextAwareTransformer):
    def leave_Param(
        self, original_node: libcst.Param, updated_node: libcst.Param
    ) -> libcst.Param:
        return updated_node.with_changes(annotation=None)

    def leave_FunctionDef(
        self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef
    ) -> libcst.FunctionDef:
        return updated_node.with_changes(returns=None)