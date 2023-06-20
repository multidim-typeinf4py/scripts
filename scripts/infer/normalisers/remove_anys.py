import libcst
from libcst import codemod, matchers as m

ANY_ = m.Attribute(m.Name("typing"), m.Name("Any")) | m.Name("Any")

class RemoveAnys(codemod.ContextAwareTransformer):
    @m.call_if_inside(m.Annotation())
    def leave_Subscript(
        self,
        original_node: libcst.Subscript,
        updated_node: libcst.Subscript
    ) -> libcst.BaseExpression:
        if all(m.matches(se.slice, m.Index(ANY_)) for se in original_node.slice):
            return updated_node.value
        else:
            return updated_node