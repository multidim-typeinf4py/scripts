import itertools
import typing

import libcst
from libcst import codemod, matchers as m


UNION_ = m.Name("Union") | m.Attribute(m.Name("typing"), m.Name("Union"))


class Unnest(codemod.ContextAwareTransformer):
    @m.call_if_inside(m.Annotation(m.Subscript(value=UNION_)))
    def leave_Subscript(
        self,
        original_node: libcst.Subscript,
        updated_node: libcst.Subscript,
    ) -> libcst.Subscript:

        flattened = list[libcst.SubscriptElement]()
        for subscript_element in updated_node.slice:
            if self.matches(subscript_element, m.SubscriptElement(m.Index(m.Subscript(UNION_)))):
                index = typing.cast(libcst.Index, subscript_element.slice)
                subscript = typing.cast(libcst.Subscript, index.value)
                flattened.extend(subscript.slice)

            else:
                flattened.append(subscript_element)

        return updated_node.with_changes(slice=flattened)


class Pep604(codemod.ContextAwareTransformer):
    @m.call_if_inside(m.Annotation())
    @m.call_if_inside(m.BinaryOperation(operator=m.BitOr()))
    def leave_BinaryOperation(
        self,
        original_node: libcst.BinaryOperation,
        updated_node: libcst.BinaryOperation,
    ) -> libcst.BaseExpression:
        return libcst.Subscript(
            value=libcst.Attribute(libcst.Name("typing"), libcst.Name("Union")),
            slice=[
                libcst.SubscriptElement(libcst.Index(updated_node.left)),
                libcst.SubscriptElement(libcst.Index(updated_node.right)),
            ],
        )

    def leave_Module(
        self,
        original_node: libcst.Module,
        updated_node: libcst.Module,
    ) -> libcst.Module:
        return updated_node.visit(Unnest(context=self.context))
