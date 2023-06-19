import libcst
from libcst import codemod

class ParametricTypeDepthReducer(codemod.ContextAwareTransformer):
    def __init__(self, context: codemod.CodemodContext, max_annot_depth=2):
        super().__init__(context=context)
        self.max_annot_depth = max_annot_depth
        self.max_annot = 0
        self.current_annot_depth = 0

    def visit_Subscript(self, node):
        self.current_annot_depth += 1
        if self.max_annot < self.current_annot_depth:
            self.max_annot += 1

    def leave_Subscript(self, original_node, updated_node):
        self.current_annot_depth -= 1
        return updated_node

    def leave_SubscriptElement(self, original_node, updated_node):
        if self.max_annot > self.max_annot_depth and self.current_annot_depth == self.max_annot_depth:
            self.max_annot -= 1
            return updated_node.with_changes(slice=libcst.Index(
                value=libcst.Attribute(libcst.Name("typing"), libcst.Name("Any"))
            ))
        else:
            return updated_node