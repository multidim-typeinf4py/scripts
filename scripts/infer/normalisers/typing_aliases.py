import typing
import libcst
from libcst import codemod, matchers as m

from scripts.infer.normalisers._matchers import DICT_ATTR_, TUPLE_ATTR_, LIST_ATTR_, SET_ATTR_, DICT_NAME, TUPLE_NAME_, \
    LIST_NAME_, SET_NAME_, TEXT_ATTR_, TEXT_NAME_, OPTIONAL_ATTR_, OPTIONAL_NAME_, FINAL_ATTR_, FINAL_NAME_


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

        elif self.matches(updated_node, SET_ATTR_):
            return libcst.Name("set")

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

        elif self.matches(updated_node, SET_NAME_):
            return libcst.Name("set")

        else:
            return updated_node


class TextToStr(codemod.ContextAwareTransformer):
    @m.call_if_inside(m.Annotation())
    def leave_Attribute(
        self,
        original_node: libcst.Attribute,
        updated_node: libcst.Attribute,
    ) -> libcst.BaseExpression:
        if self.matches(updated_node, TEXT_ATTR_):
            return libcst.Name("str")
        return updated_node

    @m.call_if_inside(m.Annotation())
    @m.call_if_not_inside(m.Attribute())
    def leave_Name(
        self,
        original_node: libcst.Name,
        updated_node: libcst.Name,
    ) -> libcst.BaseExpression:
        if self.matches(updated_node, TEXT_NAME_):
            return libcst.Name("str")
        return updated_node





class RemoveOuterOptional(codemod.ContextAwareTransformer):
    @m.call_if_inside(m.Annotation(m.Subscript(
        value=OPTIONAL_ATTR_ | OPTIONAL_NAME_, 
        slice=[m.SubscriptElement(m.Index())]
    )))
    def leave_Annotation(
        self,
        original_node: libcst.Annotation,
        updated_node: libcst.Annotation,
    ) -> libcst.Annotation:
        subscript = typing.cast(libcst.Subscript, updated_node.annotation)
        index = typing.cast(libcst.Index, subscript.slice[0].slice)
        return updated_node.with_changes(annotation=index.value)


class RemoveOuterFinal(codemod.ContextAwareTransformer):
    @m.call_if_inside(m.Annotation(m.Subscript(
        value=FINAL_ATTR_ | FINAL_NAME_, 
        slice=[m.SubscriptElement(m.Index())]
    )))
    def leave_Annotation(
        self,
        original_node: libcst.Annotation,
        updated_node: libcst.Annotation,
    ) -> libcst.Annotation:
        subscript = typing.cast(libcst.Subscript, updated_node.annotation)
        index = typing.cast(libcst.Index, subscript.slice[0].slice)
        return updated_node.with_changes(annotation=index.value)