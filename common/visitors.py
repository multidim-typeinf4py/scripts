import abc
import itertools
import typing

import libcst
from libcst import metadata, matchers as m, helpers as h

NAME = m.Name()
INSTANCE_ATTR = m.Attribute(m.Name("self"), m.Name())


class ScopeAwareVisitor(m.MatcherDecoratableVisitor):
    def __init__(self) -> None:
        super().__init__()
        self._qualifier: list[str] = []

    def qualified_name(self, name: libcst.CSTNode | str) -> str:
        name = h.get_full_name_for_node_or_raise(name)
        return ".".join((*self._qualifier, name))

    @m.visit(m.FunctionDef() | m.ClassDef())
    def __on_enter_scope(self, node: libcst.FunctionDef | libcst.ClassDef) -> None:
        self._qualifier.append(node.name.value)

    @m.leave(m.FunctionDef() | m.ClassDef())
    def __on_leave_scope(self, _: libcst.FunctionDef | libcst.ClassDef) -> None:
        self._qualifier.pop()


class HintableParameterVisitor(m.MatcherDecoratableVisitor, abc.ABC):
    @m.call_if_inside(m.Param())
    @m.visit(m.Param())
    def __on_visit_param(self, param: libcst.Param) -> None:
        if param.annotation is not None:
            self.annotated_param(param, param.annotation)
        else:
            self.unannotated_param(param)

    @abc.abstractmethod
    def annotated_param(self, param: libcst.Param, annotation: libcst.Annotation) -> None:
        ...

    @abc.abstractmethod
    def unannotated_param(self, param: libcst.Param) -> None:
        ...


class HintableReturnVisitor(m.MatcherDecoratableVisitor, abc.ABC):
    @m.call_if_inside(m.FunctionDef())
    @m.visit(m.FunctionDef())
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


class HintableDeclarationVisitor(m.MatcherDecoratableVisitor, abc.ABC):
    """
    Provide hook methods for visiting hintable attributes (both a and self.a)
    in Assign, AnnAssign and AugAssign, as well as WithItems, For Loops and Walrus usages.
    """

    @m.call_if_inside(m.AnnAssign(target=NAME | INSTANCE_ATTR))
    @m.visit(m.AnnAssign())
    def __on_visit_annassign(self, assignment: libcst.AnnAssign) -> None:
        if assignment.value is not None:
            self.annotated_assignment(assignment.target, assignment.annotation)
        else:
            self.annotated_hint(assignment.target, assignment.annotation)

    @abc.abstractmethod
    def annotated_assignment(
        self, target: libcst.Name | libcst.Attribute, annotation: libcst.Annotation
    ) -> None:
        ...

    @abc.abstractmethod
    def annotated_hint(
        self, target: libcst.Name | libcst.Attribute, annotation: libcst.Annotation
    ) -> None:
        ...

    @m.call_if_inside(
        m.AssignTarget(target=NAME | INSTANCE_ATTR | m.Tuple() | m.List())
        | m.AugAssign(target=NAME | INSTANCE_ATTR | m.Tuple() | m.List())
        | m.WithItem(asname=m.AsName(NAME | INSTANCE_ATTR | m.Tuple() | m.List()))
        | m.For(target=NAME | INSTANCE_ATTR | m.Tuple() | m.List())
        | m.CompFor(target=NAME | INSTANCE_ATTR | m.Tuple() | m.List())
        | m.NamedExpr(target=NAME | INSTANCE_ATTR | m.Tuple() | m.List())
    )
    @m.visit(
        m.AssignTarget() | m.AugAssign() | m.WithItem() | m.For() | m.CompFor() | m.NamedExpr()
    )
    def __on_visit_target(
        self,
        node: libcst.AssignTarget
        | libcst.AugAssign
        | libcst.WithItem
        | libcst.For
        | libcst.CompFor
        | libcst.NamedExpr,
    ) -> None:
        if hasattr(node, "target"):
            target = node.target
        elif hasattr(node, "asname"):
            target = node.asname.name

        if m.matches(target, NAME | INSTANCE_ATTR):
            self.unannotated_target(target)

        elif m.matches(target, m.Tuple() | m.List()):
            elements: typing.Sequence[libcst.BaseElement] = m.findall(
                target,
                m.StarredElement(NAME | INSTANCE_ATTR) | m.Element(NAME | INSTANCE_ATTR),
            )

            for element in elements:
                self.unannotated_target(element.value)

    @abc.abstractmethod
    def unannotated_target(
        self,
        target: libcst.Name | libcst.Attribute,
    ) -> None:
        ...