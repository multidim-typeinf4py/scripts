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