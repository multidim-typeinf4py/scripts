import libcst
from libcst import codemod, matchers as m


_QUALIFIED_BOOL_LIT = m.Attribute(m.Name("builtins"), m.Name("False") | m.Name("True"))
_UNQUALIFIED_BOOL_LIT = m.Name("False") | m.Name("True")

_QUALIFIED_SUBSCRIPT_LITERAL = m.Subscript(m.Attribute(m.Name("typing"), m.Name("Literal")))
_UNQUALIFIED_SUBSCRIPT_LITERAL = m.Subscript(m.Name("Literal"))

class LiteralToBaseClass(codemod.ContextAwareTransformer):
    @m.call_if_inside(m.Annotation())
    def leave_Attribute(
        self, original_node: libcst.Attribute, updated_node: libcst.Attribute
    ) -> libcst.Attribute:
        if self.matches(original_node, _QUALIFIED_BOOL_LIT):
            return libcst.Attribute(libcst.Name("builtins"), libcst.Name("bool"))

        if self.matches(original_node, _QUALIFIED_SUBSCRIPT_LITERAL):
            return libcst.Attribute(libcst.Name("builtins"), libcst.Name("Literal"))
        return updated_node

    @m.call_if_inside(m.Annotation())
    @m.call_if_not_inside(m.Attribute())
    def leave_Name(
        self, original_node: libcst.Name, updated_node: libcst.Name
    ) -> libcst.Name:
        if self.matches(original_node, _UNQUALIFIED_BOOL_LIT):
            return libcst.Name("bool")
        if self.matches(original_node, _UNQUALIFIED_SUBSCRIPT_LITERAL):
            return libcst.Name("Literal")
        return updated_node
