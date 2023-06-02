import abc
import typing
from typing import Union, Optional

import libcst
from libcst import metadata, matchers as m, helpers as h

from src.common.metadata.keyword_scopage import KeywordModifiedScopeProvider
from . import _traversal


class ScopeAwareVisitor(m.MatcherDecoratableVisitor):
    def __init__(self) -> None:
        super().__init__()
        self._qualifier: list[str] = []

    def qualified_scope(self) -> tuple[str, ...]:
        return tuple(self._qualifier)

    def qualified_name(self, name: Union[libcst.CSTNode, str]) -> str:
        name = h.get_full_name_for_node_or_raise(name)
        return ".".join((*self._qualifier, name))

    @m.visit(m.FunctionDef() | m.ClassDef())
    def __on_enter_scope(
        self, node: Union[libcst.FunctionDef, libcst.ClassDef]
    ) -> None:
        self._qualifier.append(node.name.value)

    @m.leave(m.FunctionDef() | m.ClassDef())
    def __on_leave_scope(self, _: Union[libcst.FunctionDef, libcst.ClassDef]) -> None:
        self._qualifier.pop()


class HintableParameterVisitor(m.MatcherDecoratableVisitor, abc.ABC):
    @m.visit(m.Param())
    @m.call_if_inside(m.Param())
    def __on_visit_param(self, param: libcst.Param) -> None:
        if param.annotation is not None:
            self.annotated_param(param, param.annotation)
        else:
            self.unannotated_param(param)

    @abc.abstractmethod
    def annotated_param(
        self, param: libcst.Param, annotation: libcst.Annotation
    ) -> None:
        ...

    @abc.abstractmethod
    def unannotated_param(self, param: libcst.Param) -> None:
        ...


class HintableReturnVisitor(m.MatcherDecoratableVisitor, abc.ABC):
    @m.visit(m.FunctionDef())
    @m.call_if_inside(m.FunctionDef())
    def __on_visit_function(self, function: libcst.FunctionDef) -> None:
        if function.returns is not None:
            self.annotated_function(function, function.returns)
        else:
            self.unannotated_function(function)

    @abc.abstractmethod
    def annotated_function(
        self, function: libcst.FunctionDef, annotation: libcst.Annotation
    ) -> None:
        ...

    @abc.abstractmethod
    def unannotated_function(self, function: libcst.FunctionDef) -> None:
        ...


class HintableDeclarationVisitor(
    m.MatcherDecoratableVisitor,
    _traversal.Recognition,
    _traversal.Traverser[None],
    abc.ABC,
):
    """
    Provide hook methods for visiting hintable attributes (both a and self.a)
    in Assign, AnnAssign and AugAssign, as well as WithItems and For Loops usages
    """

    METADATA_DEPENDENCIES = (
        KeywordModifiedScopeProvider,
        metadata.ScopeProvider,
    )

    @m.call_if_inside(_traversal.Matchers.annassign)
    def visit_AnnAssign_target(self, assignment: libcst.AnnAssign) -> None:
        if targets := self.instance_attribute_hint_targets(assignment):
            visitor = self.instance_attribute_hint
        elif targets := self.libsa4py_hint_targets(assignment):
            visitor = self.libsa4py_hint
        elif targets := self.annotated_hint_targets(assignment):
            visitor = self.annotated_hint
        elif targets := self.annotated_assignment_targets(assignment):
            visitor = self.annotated_assignment
        else:
            self.fallthru(assignment)
            return None

        self._apply_visit(targets, visitor, assignment)

    # @m.call_if_inside(m.CompFor(target=NAME | INSTANCE_ATTR | m.Tuple() | m.List()))
    # def visit_CompFor_target(self, node: libcst.CompFor) -> None:
    #     return self.__on_visit_target(node)

    @m.call_if_inside(_traversal.Matchers.assign)
    def visit_Assign_targets(self, node: libcst.Assign) -> None:
        if targets := self.libsa4py_hint_targets(node):
            visitor = self.libsa4py_hint

        elif targets := self.unannotated_assign_single_targets(node):
            visitor = self.unannotated_assign_single_target

        elif targets := self.unannotated_assign_multiple_targets(node):
            visitor = self.unannotated_assign_multiple_targets_or_augassign

        else:
            self.fallthru(node)
            return None

        self._apply_visit(targets, visitor, node)

    @m.call_if_inside(_traversal.Matchers.augassign)
    def visit_AugAssign_target(self, node: libcst.AugAssign) -> None:
        if targets := self.augassign_targets(node):
            visitor = self.unannotated_assign_multiple_targets_or_augassign
        else:
            self.fallthru(node)
            return None

        return self._apply_visit(targets, visitor, node)

    @m.call_if_inside(_traversal.Matchers.fortargets)
    def visit_For_target(self, node: libcst.For) -> None:
        if targets := self.for_targets(node):
            visitor = self.for_target
        else:
            self.fallthru(node)
            return None

        return self._apply_visit(targets, visitor, node)

    @m.call_if_inside(_traversal.Matchers.withitems)
    def visit_With(self, node: libcst.With) -> None:
        if targets := self.with_targets(node):
            visitor = self.withitem_target
        else:
            self.fallthru(node)
            return None

        return self._apply_visit(targets, visitor, node)

    # We cannot annotate anything inside a lambda; and annotating
    # variables from outside a Lambda is an alternation to the scope
    def visit_Lambda(self, _: libcst.Lambda) -> Optional[bool]:
        return False

    _T = typing.TypeVar("_T", bound=libcst.CSTNode)

    def _apply_visit(
        self,
        targets: _traversal.Targets,
        visitor: typing.Callable[[_T, Union[libcst.Name, libcst.Attribute]], None],
        original_node: _T,
    ) -> None:
        for target in targets.unchanged:
            visitor(original_node, target)

        for target in targets.glbls:
            self.global_target(original_node, target)

        for target in targets.nonlocals:
            self.nonlocal_target(original_node, target)
