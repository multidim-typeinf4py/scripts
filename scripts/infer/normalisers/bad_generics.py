import typing

import libcst
from libcst import codemod, matchers as m


TUPLE_ = libcst.Attribute(libcst.Name("typing"), libcst.Name("Tuple"))
LIST_ = libcst.Attribute(libcst.Name("typing"), libcst.Name("List"))
DICT_ = libcst.Attribute(libcst.Name("typing"), libcst.Name("Dict"))
UNION_ = libcst.Attribute(libcst.Name("typing"), libcst.Name("Union"))


_QUALIFIED_BOOL_LIT = m.Attribute(m.Name("builtins"), m.Name("False") | m.Name("True"))
_UNQUALIFIED_BOOL_LIT = m.Name("False") | m.Name("True")

_QUALIFIED_SUBSCRIPT_LITERAL = m.Subscript(
    m.Attribute(m.Name("typing"), m.Name("Literal"))
)
_UNQUALIFIED_SUBSCRIPT_LITERAL = m.Subscript(m.Name("Literal"))


class BadGenericsNormaliser(codemod.ContextAwareTransformer):
    @m.call_if_inside(m.Annotation())
    def leave_Tuple(
        self,
        original_node: libcst.Tuple,
        updated_node: libcst.Tuple,
    ) -> libcst.BaseExpression:
        if len(updated_node.elements) == 0:
            return TUPLE_

        return libcst.Subscript(
            value=TUPLE_,
            slice=_replace_elements(updated_node.elements),
        )

    @m.leave(m.Annotation(m.List()))
    def leave_outer_list_anno(
        self,
        original_node: libcst.Annotation,
        updated_node: libcst.Annotation,
    ) -> libcst.Annotation:
        list_ = typing.cast(libcst.List, updated_node.annotation)
        if len(list_.elements) == 0:
            return libcst.Annotation(LIST_)

        elif len(list_.elements) > 1:
            list_typing = [
                libcst.SubscriptElement(
                    libcst.Index(
                        libcst.Subscript(
                            value=UNION_,
                            slice=_replace_elements(list_.elements),
                        )
                    )
                )
            ]

        else:
            list_typing = _replace_elements(list_.elements)

        return libcst.Annotation(
            libcst.Subscript(
                value=LIST_,
                slice=list_typing,
            )
        )

    @m.call_if_inside(m.Annotation())
    @m.leave(m.Dict(elements=[]))
    def leave_outer_dict_anno(
        self,
        original_node: libcst.Dict,
        updated_node: libcst.Dict,
    ) -> libcst.BaseExpression:
        return DICT_

    @m.call_if_inside(m.Annotation())
    def leave_Attribute(
        self, original_node: libcst.Attribute, updated_node: libcst.Attribute
    ) -> libcst.Attribute:
        if self.matches(original_node, _QUALIFIED_BOOL_LIT | _UNQUALIFIED_BOOL_LIT):
            return libcst.Attribute(libcst.Name("builtins"), libcst.Name("bool"))

        if self.matches(original_node, _QUALIFIED_SUBSCRIPT_LITERAL | _UNQUALIFIED_SUBSCRIPT_LITERAL):
            return libcst.Attribute(libcst.Name("builtins"), libcst.Name("Literal"))
        return updated_node


def _replace_elements(
    elements: typing.Sequence[libcst.BaseElement],
) -> list[libcst.SubscriptElement]:
    assert all(map(lambda e: isinstance(e, libcst.Element), elements))
    return [libcst.SubscriptElement(libcst.Index(value=e.value)) for e in elements]
