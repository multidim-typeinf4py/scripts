import libcst
from libcst import codemod, matchers as m

from libcst import ParserSyntaxError

class Unquote(codemod.ContextAwareTransformer):
    @m.call_if_inside(m.Annotation())
    def leave_SimpleString(
        self, original_node: libcst.SimpleString, updated_node: libcst.SimpleString
    ) -> libcst.BaseExpression:
        try:
            unquoted = libcst.parse_expression(updated_node.value[1:-1])
            return unquoted
        except ParserSyntaxError:
            return original_node

    @m.call_if_inside(m.Annotation())
    @m.leave(m.Subscript(m.Name("Annotated") | m.Attribute(m.Name(), m.Name("Annotated"))))
    def _convert_annotated(self, original_node: libcst.Subscript, updated_node: libcst.Subscript) -> libcst.BaseExpression:
        """Docs: Add metadata x to a given type T by using the annotation Annotated[T, x]"""
        if self.matches(updated_node.slice[0], m.SubscriptElement(m.Index())):
            return updated_node.slice[0].slice.value
        return updated_node
