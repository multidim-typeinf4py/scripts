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

    def qualified_scope(self) -> tuple[str, ...]:
        return tuple(self._qualifier)

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
    @m.visit(m.Param())
    @m.call_if_inside(m.Param())
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


class HintableDeclarationVisitor(m.MatcherDecoratableVisitor, abc.ABC):
    """
    Provide hook methods for visiting hintable attributes (both a and self.a)
    in Assign, AnnAssign and AugAssign, as well as WithItems, For Loops and Walrus usages.
    """

    @abc.abstractmethod
    def instance_attribute_hint(
        self, target: libcst.Name, annotation: libcst.Annotation | None
    ) -> None:
        """
        class C:
            a: int      # triggers
            a = ...     # triggers (libsa4py's instance attributes)
            a = 5       # ignored

        a: int          # ignored
        """
        ...

    @abc.abstractmethod
    def annotated_assignment(
        self, target: libcst.Name | libcst.Attribute, annotation: libcst.Annotation
    ) -> None:
        """
        a: int = 5      # triggers
        a: int          # ignored
        """
        ...

    @abc.abstractmethod
    def annotated_hint(
        self, target: libcst.Name | libcst.Attribute, annotation: libcst.Annotation
    ) -> None:
        """
        a: int = 5      # ignored
        a: int          # triggers

        class C:
            a: int      # ignored
        """
        ...

    @abc.abstractmethod
    def unannotated_target(
        self,
        target: libcst.Name | libcst.Attribute,
    ) -> None:
        """
        class C:
            a: int      # ignored
            a = ...     # ignored
            a = 5       # triggers

        a = 10          # triggers
        a: int          # ignored

        a = b = 50          # triggers for both a and b

        for x, y in zip([1, 2, 3], "abc"): # triggers for both x and y
            ...

        with p.open() as f:     # triggers for f
            ...

        [x.value for x in y]    # triggers for x

        assert (x := 10) >= 5   # triggers for x
        """
        ...

    # @m.visit(m.AnnAssign())
    @m.call_if_inside(m.AnnAssign(target=NAME | INSTANCE_ATTR))
    def visit_AnnAssign_target(self, assignment: libcst.AnnAssign) -> None:
        if assignment.value is not None:
            self.annotated_assignment(assignment.target, assignment.annotation)
        elif isinstance(self.get_metadata(metadata.ScopeProvider, assignment), metadata.ClassScope):
            self.instance_attribute_hint(assignment.target, assignment.annotation)
        else:
            self.annotated_hint(assignment.target, assignment.annotation)

    # Catch libsa4py's retainment of INSTANCE_ATTRs
    @m.call_if_inside(m.ClassDef())
    def visit_ClassDef_body(self, clazz: libcst.ClassDef) -> None:
        for libsa4py_hint in filter(
            lambda e: m.matches(
                e,
                m.SimpleStatementLine(
                    body=[m.Assign(targets=[m.AssignTarget(target=m.Name())], value=m.Ellipsis())]
                ),
            ),
            clazz.body.body,
        ):
            self.instance_attribute_hint(libsa4py_hint.body[0].targets[0].target, annotation=None)

    @m.call_if_inside(m.For(target=NAME | INSTANCE_ATTR | m.Tuple() | m.List()))
    def visit_For_target(self, node: libcst.For) -> None:
        return self.__on_visit_target(node)

    @m.call_if_inside(m.CompFor(target=NAME | INSTANCE_ATTR | m.Tuple() | m.List()))
    def visit_CompFor_target(self, node: libcst.CompFor) -> None:
        return self.__on_visit_target(node)

    @m.call_if_inside(
        m.Assign(
            targets=[
                m.ZeroOrMore(m.AssignTarget(target=NAME | INSTANCE_ATTR | m.Tuple() | m.List()))
            ],
            value=~m.Ellipsis(),
        )
    )
    def visit_Assign_targets(self, node: libcst.Assign) -> None:
        for target in node.targets:
            if m.matches(target.target, NAME | INSTANCE_ATTR | m.Tuple() | m.List()):
                self.__on_visit_target(target)

    @m.call_if_inside(m.AugAssign(target=NAME | INSTANCE_ATTR | m.Tuple() | m.List()))
    def visit_AugAssign_target(self, node: libcst.AugAssign) -> None:
        return self.__on_visit_target(node)

    @m.call_if_inside(m.WithItem(asname=m.AsName(NAME | INSTANCE_ATTR | m.Tuple() | m.List())))
    def visit_WithItem_item(self, node: libcst.WithItem) -> None:
        return self.__on_visit_target(node)

    @m.call_if_inside(m.NamedExpr(target=NAME | INSTANCE_ATTR | m.Tuple() | m.List()))
    def visit_NamedExpr_target(self, node: libcst.NamedExpr) -> None:
        return self.__on_visit_target(node)

    def __on_visit_target(
        self,
        node: libcst.For
        | libcst.CompFor
        | libcst.AssignTarget
        | libcst.AugAssign
        | libcst.WithItem
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
                (m.StarredElement | m.Element)(NAME | INSTANCE_ATTR),
            )

            for element in elements:
                self.unannotated_target(element.value)
