import typing

import libcst
from libcst import codemod, matchers as m

from ._matchers import LIST_, UNION_, TUPLE_, _QUALIFIED_BOOL_LIT, _UNQUALIFIED_BOOL_LIT, DICT_, \
    _QUALIFIED_SUBSCRIPT_LITERAL, _UNQUALIFIED_SUBSCRIPT_LITERAL


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
                            value=libcst.Attribute(libcst.Name("typing"), libcst.Name("Union")),
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
        return updated_node

    @m.call_if_inside(m.Annotation(_QUALIFIED_SUBSCRIPT_LITERAL | _UNQUALIFIED_SUBSCRIPT_LITERAL))
    @m.leave(_QUALIFIED_SUBSCRIPT_LITERAL | _UNQUALIFIED_SUBSCRIPT_LITERAL)
    def leave_subscript_literal(
        self, original_node: libcst.Subscript, updated_node: libcst.Subscript
    ) -> libcst.Attribute:
        if len(original_node.slice) != 1:
            return libcst.Attribute(libcst.Name("builtins"), libcst.Name("Literal"))

        if self.matches(original_node.slice[0], m.SubscriptElement(
            m.Index(m.SimpleString())
        )):
            simple_string = typing.cast(libcst.SimpleString, updated_node.slice[0].slice.value)
            if simple_string.value.startswith("b"):
                return libcst.Attribute(libcst.Name("builtins"), libcst.Name("bytes"))

            return libcst.Attribute(libcst.Name("builtins"), libcst.Name("str"))

        elif self.matches(original_node.slice[0], m.SubscriptElement(
            m.Index(m.Integer())
        )):
            return libcst.Attribute(libcst.Name("builtins"), libcst.Name("int"))

        elif self.matches(original_node.slice[0], m.SubscriptElement(
            m.Index(m.Float())
        )):
            return libcst.Attribute(libcst.Name("builtins"), libcst.Name("float"))

        elif self.matches(original_node.slice[0], m.SubscriptElement(
            m.Index(m.Name("None"))
        )):
            return libcst.Name("None")

        from scripts.common.ast_helper import _stringify
        raise NotImplementedError(f"Unknown subscript type: {_stringify(updated_node)}")


def _replace_elements(
    elements: typing.Sequence[libcst.BaseElement],
) -> list[libcst.SubscriptElement]:
    assert all(map(lambda e: isinstance(e, libcst.Element), elements))
    return [libcst.SubscriptElement(libcst.Index(value=e.value)) for e in elements]
