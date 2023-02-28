import abc
import dataclasses
import functools
import typing

import libcst
from libcst import metadata, matchers as m, helpers as h

NAME = m.Name()
INSTANCE_ATTR = m.Attribute(m.Name("self"), m.Name())


class ScopeAwareTransformer(m.MatcherDecoratableTransformer):
    def __init__(self) -> None:
        super().__init__()
        self._qualifier: list[str] = []

    def qualified_scope(self) -> tuple[str, ...]:
        return tuple(self._qualifier)

    @m.visit(m.FunctionDef() | m.ClassDef())
    def __on_enter_scope(self, node: libcst.FunctionDef | libcst.ClassDef) -> None:
        self._qualifier.append(node.name.value)

    @m.leave(m.FunctionDef() | m.ClassDef())
    def __on_leave_scope(self, _: libcst.FunctionDef | libcst.ClassDef) -> None:
        self._qualifier.pop()


class HintableParameterTransformer(m.MatcherDecoratableTransformer, abc.ABC):
    def leave_Param(
        self, _: libcst.Param, updated_node: libcst.Param
    ) -> libcst.Param | libcst.MaybeSentinel | libcst.FlattenSentinel[
        libcst.Param
    ] | libcst.RemovalSentinel:
        if updated_node.annotation is not None:
            self.annotated_param(updated_node, updated_node.annotation)
        else:
            self.unannotated_param(updated_node)

    @abc.abstractmethod
    def annotated_param(
        self, param: libcst.Param, annotation: libcst.Annotation
    ) -> libcst.Param | libcst.MaybeSentinel | libcst.FlattenSentinel[
        libcst.Param
    ] | libcst.RemovalSentinel:
        ...

    @abc.abstractmethod
    def unannotated_param(
        self, param: libcst.Param
    ) -> libcst.Param | libcst.MaybeSentinel | libcst.FlattenSentinel[
        libcst.Param
    ] | libcst.RemovalSentinel:
        ...


class HintableReturnTransformer(m.MatcherDecoratableTransformer, abc.ABC):
    def leave_FunctionDef(
        self, function: libcst.FunctionDef
    ) -> libcst.BaseStatement | libcst.FlattenSentinel[
        libcst.BaseStatement
    ] | libcst.RemovalSentinel:
        if function.returns is not None:
            self.annotated_function(function, function.returns)
        else:
            self.unannotated_function(function)

    @abc.abstractmethod
    def annotated_function(
        self, function: libcst.FunctionDef, annotation: libcst.Annotation
    ) -> libcst.BaseStatement | libcst.FlattenSentinel[
        libcst.BaseStatement
    ] | libcst.RemovalSentinel:
        ...

    @abc.abstractmethod
    def unannotated_function(
        self, function: libcst.FunctionDef
    ) -> libcst.BaseStatement | libcst.FlattenSentinel[
        libcst.BaseStatement
    ] | libcst.RemovalSentinel:
        ...


@dataclasses.dataclass
class Prepend:
    node: libcst.CSTNode


@dataclasses.dataclass
class Append:
    node: libcst.CSTNode


@dataclasses.dataclass
class Replace:
    node: libcst.CSTNode


@dataclasses.dataclass
class Untouched:
    ...


class HintableDeclarationTransformer(m.MatcherDecoratableTransformer, abc.ABC):
    """
    Provide hook methods for transforming hintable attributes (both a and self.a)
    in Assign, AnnAssign and AugAssign, as well as WithItems, For Loops and Walrus usages.
    """

    @abc.abstractmethod
    def instance_attribute_hint(
        self, assignment: libcst.AnnAssign, target: libcst.Name, annotation: libcst.Annotation
    ) -> libcst.BaseAssignTargetExpression | libcst.FlattenSentinel[
        libcst.BaseAssignTargetExpression
    ] | libcst.RemovalSentinel:
        """
        class C:
            a: int      # triggers
            a = ...     # ignored (libsa4py's instance attributes)
            a = 5       # ignored

        a: int          # ignored
        """
        ...

    @abc.abstractmethod
    def libsa4py_hint(
        self, assignment: libcst.Assign, target: libcst.Name
    ) -> libcst.BaseAssignTargetExpression | libcst.FlattenSentinel[
        libcst.BaseAssignTargetExpression
    ] | libcst.RemovalSentinel:
        """
        class C:
            a: int      # ignored
            a = ...     # triggers (libsa4py's instance attributes)
            a = 5       # ignored

        a: int          # ignored
        """
        ...

    @abc.abstractmethod
    def annotated_assignment(
        self,
        assignment: libcst.AnnAssign,
        target: libcst.Name | libcst.Attribute,
        annotation: libcst.Annotation,
    ) -> libcst.BaseAssignTargetExpression | libcst.FlattenSentinel[
        libcst.BaseAssignTargetExpression
    ] | libcst.RemovalSentinel:
        """
        a: int = 5      # triggers
        a: int          # ignored
        """
        ...

    @abc.abstractmethod
    def annotated_hint(
        self,
        assignment: libcst.AnnAssign,
        target: libcst.Name | libcst.Attribute,
        annotation: libcst.Annotation,
    ) -> libcst.BaseAssignTargetExpression | libcst.FlattenSentinel[
        libcst.BaseAssignTargetExpression
    ] | libcst.RemovalSentinel:
        """
        a: int = 5      # ignored
        a: int          # triggers

        class C:
            a: int      # ignored
        """
        ...

    @abc.abstractmethod
    def unannotated_assign_single_target(
        self,
        assign: libcst.Assign,
        target: libcst.Name | libcst.Attribute,
    ) -> libcst.BaseAssignTargetExpression | libcst.FlattenSentinel[
        libcst.BaseAssignTargetExpression
    ] | libcst.RemovalSentinel:
        """
        class C:
            a: int      # ignored
            a = ...     # ignored
            a = 5       # triggers

        a = 10          # triggers
        a: int          # ignored

        a = b = 50          # ignored

        for x, y in zip([1, 2, 3], "abc"): # triggers for both x and y
            ...

        with p.open() as f:     # triggers for f
            ...

        [x.value for x in y]    # triggers for x

        assert (x := 10) >= 5   # triggers for x
        """
        ...

    @abc.abstractmethod
    def unannotated_assign_multiple_target(
        self,
        assign: libcst.Assign,
        target: libcst.Name | libcst.Attribute,
    ) -> libcst.BaseAssignTargetExpression | libcst.FlattenSentinel[
        libcst.BaseAssignTargetExpression
    ] | libcst.RemovalSentinel:
        """
        a = b = 50      # triggers
        """
        ...

    @m.call_if_inside(m.AnnAssign(target=NAME | INSTANCE_ATTR))
    def visit_AnnAssign(
        self, assignment: libcst.AnnAssign
    ) -> libcst.BaseAssignTargetExpression | libcst.FlattenSentinel[
        libcst.BaseAssignTargetExpression
    ] | libcst.RemovalSentinel:
        if assignment.value is not None:
            return self.annotated_assignment(assignment, assignment.target, assignment.annotation)
        elif isinstance(self.get_metadata(metadata.ScopeProvider, assignment), metadata.ClassScope):
            return self.instance_attribute_hint(
                assignment, assignment.target, assignment.annotation
            )
        else:
            return self.annotated_hint(assignment, assignment.target, assignment.annotation)

    # Catch libsa4py's retainment of INSTANCE_ATTRs
    @m.call_if_inside(m.ClassDef())
    def visit_ClassDef(self, clazz: libcst.ClassDef) -> None:
        for libsa4py_hint in filter(
            lambda e: m.matches(
                e,
                m.SimpleStatementLine(
                    body=[m.Assign(targets=[m.AssignTarget(target=m.Name())], value=m.Ellipsis())]
                ),
            ),
            clazz.body.body,
        ):
            assign = libsa4py_hint.body[0]
            target = assign.targets[0].target
            self.libsa4py_hint(assign, target, annotation=None)

    @abc.abstractmethod
    def for_target(
        self, forloop: libcst.For, target: libcst.Name | libcst.Attribute
    ) -> libcst.BaseAssignTargetExpression | libcst.FlattenSentinel[
        libcst.BaseAssignTargetExpression
    ] | libcst.RemovalSentinel:
        ...

    def leave_For(
        self, _: libcst.For, updated_node: libcst.For
    ) -> libcst.BaseStatement | libcst.FlattenSentinel[
        libcst.BaseStatement
    ] | libcst.RemovalSentinel:
        return super().leave_For(original_node, updated_node)

    @m.call_if_inside(m.For(target=NAME | INSTANCE_ATTR | m.Tuple() | m.List()))
    def visit_For(self, node: libcst.For) -> None:
        return libcst.FlattenSentinel()

    @m.call_if_inside(m.CompFor(target=NAME | INSTANCE_ATTR | m.Tuple() | m.List()))
    def visit_CompFor(self, node: libcst.CompFor) -> None:
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

    def _unpack_targets(
        self, node: libcst.Name | libcst.Attribute | libcst.Tuple | libcst.List
    ) -> typing.Generator[libcst.BaseAssignTargetExpression]:
        if m.matches(node, NAME | INSTANCE_ATTR):
            yield node

        yield from (
            element.value
            for element in m.findall(node, (m.StarredElement | m.Element)(NAME | INSTANCE_ATTR))
        )
