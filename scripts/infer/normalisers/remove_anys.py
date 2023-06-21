import libcst
from libcst import codemod, matchers as m

from ._matchers import ANY_


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