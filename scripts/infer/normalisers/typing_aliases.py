import libcst
from libcst import codemod, matchers as m

DICT_ATTR_ = m.Attribute(m.Name("typing"), m.Name("Dict"))
DICT_NAME = m.Name("Dict")

TUPLE_ATTR_ = m.Attribute(m.Name("typing"), m.Name("Tuple"))
TUPLE_NAME_ = m.Name("Tuple")

LIST_ATTR_ = m.Attribute(m.Name("typing"), m.Name("List"))
LIST_NAME_ = m.Name("List")


class LowercaseTypingAliases(codemod.ContextAwareTransformer):
    @m.call_if_inside(m.Annotation())
    def leave_Attribute(
        self,
        original_node: libcst.Attribute,
        updated_node: libcst.Attribute,
    ) -> libcst.BaseExpression:
        if self.matches(updated_node, DICT_ATTR_):
            return libcst.Name("dict")

        elif self.matches(updated_node, TUPLE_ATTR_):
            return libcst.Name("tuple")

        elif self.matches(updated_node, LIST_ATTR_):
            return libcst.Name("list")

        else:
            return updated_node

    @m.call_if_inside(m.Annotation())
    @m.call_if_not_inside(m.Attribute())
    def leave_Name(
        self,
        original_node: libcst.Name,
        updated_node: libcst.Name,
    ) -> libcst.BaseExpression:
        if self.matches(updated_node, DICT_NAME):
            return libcst.Name("dict")

        elif self.matches(updated_node, TUPLE_NAME_):
            return libcst.Name("tuple")

        elif self.matches(updated_node, LIST_NAME_):
            return libcst.Name("list")

        else:
            return updated_node
