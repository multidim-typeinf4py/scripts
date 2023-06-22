import typing

import libcst
from libcst import codemod, matchers as m

from scripts.common import _stringify

from ._matchers import UNION_, OPTIONAL_


class UnionNormaliser(codemod.Codemod):
    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        optional_to_union = _OptionalToUnion(context=self.context).transform_module(tree)
        union_or_to_union_t = _Pep604(context=self.context).transform_module(optional_to_union)
        flattened = _Flatten(context=self.context).transform_module(union_or_to_union_t)
        sorted = _UnionSorter(context=self.context).transform_module(flattened)

        return sorted


class _OptionalToUnion(codemod.ContextAwareTransformer):
    @m.call_if_inside(m.Annotation())
    def leave_Subscript(
        self, original_node: libcst.Subscript, updated_node: libcst.Subscript
    ) -> libcst.Subscript:
        if not m.matches(original_node.value, OPTIONAL_):
            return updated_node
        optional_t = original_node.slice[0]
        rewritten_subscript = libcst.Subscript(
            value=libcst.Attribute(libcst.Name("typing"), libcst.Name("Union")),
            slice=[
                optional_t,
                libcst.SubscriptElement(libcst.Index(libcst.Name("None"))),
            ],
        )
        return rewritten_subscript



class _Pep604(codemod.ContextAwareTransformer):
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


class _Flatten(codemod.ContextAwareTransformer):
    @m.call_if_inside(m.Annotation(m.Subscript(value=UNION_)))
    def leave_Subscript(
        self,
        original_node: libcst.Subscript,
        updated_node: libcst.Subscript,
    ) -> libcst.Subscript:
        if not m.matches(updated_node, m.Subscript(value=UNION_)):
            return updated_node

        flattened = list[libcst.SubscriptElement]()
        for subscript_element in updated_node.slice:
            # Union element with further inner types
            if self.matches(subscript_element, m.SubscriptElement(m.Index(m.Subscript(UNION_)))):
                index = typing.cast(libcst.Index, subscript_element.slice)
                subscript = typing.cast(libcst.Subscript, index.value)
                flattened.extend(subscript.slice)

            # Union element without further types
            elif self.matches(subscript_element, m.SubscriptElement(m.Index(UNION_))):
                # no inner types to read from
                continue

            else:
                flattened.append(subscript_element)

        return updated_node.with_changes(
            value=libcst.Attribute(libcst.Name("typing"), libcst.Name("Union")),
            slice=flattened,
        )


class _UnionSorter(codemod.ContextAwareTransformer):
    @m.call_if_inside(m.Annotation(m.Subscript(value=UNION_)))
    def leave_Subscript(
        self,
        original_node: libcst.Subscript,
        updated_node: libcst.Subscript,
    ) -> libcst.Subscript:
        subscript_elems_as_str = set(_stringify(se.slice.value) for se in updated_node.slice)
        as_sorted = sorted(subscript_elems_as_str)

        sorted_union = libcst.parse_expression(f"typing.Union[{', '.join(as_sorted)}]")

        return typing.cast(libcst.Subscript, sorted_union)
