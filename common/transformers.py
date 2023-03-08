import abc
import dataclasses
import itertools

import libcst
from libcst import metadata, matchers as m, helpers as h, codemod as c

from .matchers import NAME, INSTANCE_ATTR, LIST, TUPLE

from common.metadata import KeywordModifiedScopeProvider, KeywordContext


class ScopeAwareTransformer(c.ContextAwareTransformer):
    def __init__(self, context: c.CodemodContext) -> None:
        super().__init__(context)
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
    def __on_leave_scope(
        self, _1: libcst.FunctionDef | libcst.ClassDef, _2: libcst.FunctionDef | libcst.ClassDef
    ) -> libcst.FunctionDef | libcst.ClassDef:
        self._qualifier.pop()
        return _2


class HintableParameterTransformer(c.ContextAwareTransformer, abc.ABC):
    def leave_Param(
        self, _: libcst.Param, updated_node: libcst.Param
    ) -> libcst.Param | libcst.MaybeSentinel | libcst.FlattenSentinel[
        libcst.Param
    ] | libcst.RemovalSentinel:
        if updated_node.annotation is not None:
            return self.annotated_param(updated_node, updated_node.annotation)
        else:
            return self.unannotated_param(updated_node)

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


class HintableReturnTransformer(c.ContextAwareTransformer, abc.ABC):
    def leave_FunctionDef(
        self, _: libcst.FunctionDef, updated_node: libcst.FunctionDef
    ) -> libcst.BaseStatement | libcst.FlattenSentinel[
        libcst.BaseStatement
    ] | libcst.RemovalSentinel:
        if updated_node.returns is not None:
            return self.annotated_function(updated_node, updated_node.returns)
        else:
            return self.unannotated_function(updated_node)

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


@dataclasses.dataclass(frozen=True)
class Prepend:
    node: libcst.BaseSmallStatement


@dataclasses.dataclass(frozen=True)
class Append:
    node: libcst.BaseSmallStatement


@dataclasses.dataclass(frozen=True)
class Replace:
    matcher: m.BaseMatcherNode
    replacement: libcst.CSTNode


@dataclasses.dataclass(frozen=True)
class Untouched:
    ...


Actions = list[Untouched | Prepend | Append | Replace]


def _apply_actions(
    actions: Actions,
    updated_node: libcst.BaseSmallStatement | libcst.BaseStatement,
) -> libcst.FlattenSentinel:
    prepends = []
    appends = []

    for action in actions:
        match action:
            case Untouched():
                ...

            case Prepend(node):
                prepends.append(node)

            case Append(node):
                appends.append(node)

            case Replace(matcher, replacement):
                updated_node = m.replace(updated_node, matcher, replacement)

    # module = h.parse_template_module(
    #     "{ps}\n{un}\n{ap}",
    #     ps=libcst.SimpleStatementLine(body=prepends),
    #     un=updated_node,
    #     ap=libcst.SimpleStatementLine(body=appends),
    # )

    if isinstance(updated_node, libcst.BaseSmallStatement):
        # Must return libcst.FlattenSentinel[BaseSmallStatement]
        return libcst.FlattenSentinel((
            *prepends,
            #libcst.EmptyLine(),
            updated_node,
            #libcst.EmptyLine(),
            *appends,
        ))
    else:
        # Must return libcst.FlattenSentinel[BaseStatement]
        return libcst.FlattenSentinel((
            libcst.SimpleStatementLine(body=prepends),
            updated_node,
            libcst.SimpleStatementLine(body=appends)
        ))


class HintableDeclarationTransformer(c.ContextAwareTransformer, abc.ABC):
    """
    Provide hook methods for transforming hintable attributes (both a and self.a)
    in Assign, AnnAssign and AugAssign, as well as WithItems, For Loops
    """

    METADATA_DEPENDENCIES = (
        metadata.ParentNodeProvider,
        metadata.ScopeProvider,
        KeywordModifiedScopeProvider,
    )

    @abc.abstractmethod
    def instance_attribute_hint(
        self, updated_node: libcst.AnnAssign, target: libcst.Name
    ) -> Actions:
        """
        class C:
            a: int      # triggers
            a = ...     # ignored (libsa4py's instance attributes)
            a = 5       # ignored

        a: int          # ignored
        """
        ...

    @abc.abstractmethod
    def libsa4py_hint(self, updated_node: libcst.Assign, target: libcst.Name) -> Actions:
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
        updated_node: libcst.AnnAssign,
        target: libcst.Name | libcst.Attribute,
    ) -> Actions:
        """
        a: int = 5      # triggers
        a: int          # ignored
        """
        ...

    @abc.abstractmethod
    def annotated_hint(
        self,
        updated_node: libcst.AnnAssign,
        target: libcst.Name | libcst.Attribute,
    ) -> Actions:
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
        updated_node: libcst.Assign | libcst.AugAssign,
        target: libcst.Name | libcst.Attribute,
    ) -> Actions:
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
    def unannotated_assign_multiple_targets(
        self,
        updated_node: libcst.Assign | libcst.AugAssign,
        target: libcst.Name | libcst.Attribute,
    ) -> Actions:
        """
        a = b = 50      # triggers
        """
        ...

    @abc.abstractmethod
    def for_target(
        self, updated_node: libcst.For, target: libcst.Name | libcst.Attribute
    ) -> Actions:
        ...

    # @abc.abstractmethod
    # def compfor_target(
    #    self, updated_node: libcst.CompFor, target: libcst.Name | libcst.Attribute
    # ) -> Actions:
    #    ...

    @abc.abstractmethod
    def withitem_target(
        self,
        updated_node: libcst.With,
        target: libcst.Name | libcst.Attribute,
    ) -> Actions:
        ...

    @abc.abstractmethod
    def global_target(
        self, updated_node: libcst.Assign | libcst.AnnAssign | libcst.AugAssign, target: libcst.Name
    ) -> Actions:
        ...

    @abc.abstractmethod
    def nonlocal_target(
        self, updated_node: libcst.Assign | libcst.AnnAssign | libcst.AugAssign, target: libcst.Name
    ) -> Actions:
        ...

    @m.call_if_inside(m.AnnAssign(target=NAME | INSTANCE_ATTR))
    def leave_AnnAssign(
        self, original_node: libcst.AnnAssign, updated_node: libcst.AnnAssign
    ) -> libcst.FlattenSentinel[libcst.BaseSmallStatement]:
        if isinstance(
            self.get_metadata(metadata.ScopeProvider, original_node.target), metadata.ClassScope
        ) and m.matches(updated_node.value, m.Ellipsis()):
            transformer = self.instance_attribute_hint

        elif updated_node.value is not None:
            transformer = self.annotated_assignment

        else:
            transformer = self.annotated_hint

        targets = self._access_targets(original_node.target)
        actions = list(
            itertools.chain.from_iterable(transformer(original_node, target) for target in targets)
        )

        return _apply_actions(actions, updated_node)

    @m.call_if_inside(
        m.Assign(
            targets=[m.AtLeastN(m.AssignTarget(target=NAME | INSTANCE_ATTR | LIST | TUPLE), n=1)]
        )
    )
    def leave_Assign(
        self, original_node: libcst.Assign, updated_node: libcst.Assign
    ) -> libcst.FlattenSentinel[libcst.BaseSmallStatement]:
        # Catch libsa4py's retainment of INSTANCE_ATTRs
        if m.matches(
            original_node, m.Assign(targets=[m.AssignTarget(target=NAME)], value=m.Ellipsis())
        ) and isinstance(
            self.get_metadata(metadata.ScopeProvider, original_node.targets[0].target),
            metadata.ClassScope,
        ):
            targets = [original_node.targets[0].target]
            transformer = self.libsa4py_hint

        elif len(original_node.targets) == 1 and not m.matches(
            asstarget := original_node.targets[0], m.AssignTarget(LIST | TUPLE)
        ):
            targets = self._access_targets(asstarget.target)
            transformer = self.unannotated_assign_single_target

        else:
            targets = list(
                itertools.chain.from_iterable(
                    self._access_targets(asstarget.target) for asstarget in original_node.targets
                )
            )
            transformer = self.unannotated_assign_multiple_targets

        actions = list(
            itertools.chain.from_iterable(transformer(original_node, target) for target in targets)
        )
        return _apply_actions(actions, updated_node)

    @m.call_if_inside(m.AugAssign(target=NAME | INSTANCE_ATTR | TUPLE | LIST))
    def leave_AugAssign(
        self, original_node: libcst.AugAssign, updated_node: libcst.AugAssign
    ) -> libcst.FlattenSentinel[libcst.BaseSmallStatement]:
        targets = self._access_targets(original_node.target)
        actions = list(
            itertools.chain.from_iterable(
                self.unannotated_assign_multiple_targets(original_node, target)
                for target in targets
            )
        )

        return _apply_actions(actions, updated_node)

    # @abc.abstractmethod
    # def namedexpr_target(
    #     self, walrus: libcst.NamedExpr, target: libcst.Name | libcst.Attribute
    # ) -> Actions:
    #     ...

    @m.call_if_inside(m.For(target=NAME | INSTANCE_ATTR | TUPLE | LIST))
    def leave_For(
        self, original_node: libcst.For, updated_node: libcst.For
    ) -> libcst.FlattenSentinel[libcst.BaseSmallStatement]:
        targets = self._access_targets(original_node.target)
        actions = list(
            itertools.chain.from_iterable(
                self.for_target(original_node, target) for target in targets
            )
        )
        return _apply_actions(actions, updated_node)

    # @m.call_if_inside(m.CompFor(target=NAME | INSTANCE_ATTR | TUPLE | LIST))
    # def leave_CompFor(
    #     self, _: libcst.CompFor, updated_node: libcst.CompFor
    # ) -> libcst.FlattenSentinel[libcst.CSTNode]:
    #     targets = self._access_targets(updated_node.target)
    #     return _apply_actions(updated_node, self.compfor_target, targets)

    @m.call_if_inside(
        m.With(
            items=[
                m.AtLeastN(m.WithItem(asname=m.AsName(NAME | INSTANCE_ATTR | TUPLE | LIST)), n=1)
            ]
        )
    )
    def leave_With(
        self, original_node: libcst.With, updated_node: libcst.With
    ) -> libcst.FlattenSentinel[libcst.CSTNode]:
        targets = list(
            itertools.chain.from_iterable(
                self._access_targets(item.asname.name)
                for item in original_node.items
                if item.asname is not None
            )
        )

        actions = list(
            itertools.chain.from_iterable(
                self.withitem_target(original_node, target) for target in targets
            )
        )

        return _apply_actions(actions, updated_node)

    # TODO: Can the prependable / appendable node for NamedExpr be found by
    # TODO: using m.StatementLine(m.AtLeastN(m.NamedExpr(...), n=1))?
    # @m.call_if_inside(m.NamedExpr(target=NAME | INSTANCE_ATTR | TUPLE | LIST))
    # def leave_NamedExpr(self, updated_node: libcst.NamedExpr) -> None:
    #    targets_of_interest = self._access_targets(updated_node.target)
    #    actions = list(
    #        itertools.chain.from_iterable(
    #            self.namedexpr_target(updated_node, target) for target in targets_of_interest
    #        )
    #    )
    #    return _handle_actions(updated_node, actions)

    # Visitors ignore Lambdas, so Transformer should too
    def visit_Lambda(self, _: libcst.Lambda) -> bool | None:
        return False

    def _access_targets(
        self,
        target: libcst.Name | libcst.Attribute | libcst.List | libcst.Tuple,
    ) -> list[libcst.Name | libcst.Attribute]:
        if m.matches(target, NAME | INSTANCE_ATTR):
            targets = [target]

        elif m.matches(target, TUPLE | LIST):
            targets = [
                element.value
                for element in m.findall(
                    target,
                    (m.StarredElement | m.Element)(NAME | INSTANCE_ATTR),
                )
            ]

        else:
            targets = []

        non_overwritten_targets = list()
        for target in targets:
            modification = self.get_metadata(KeywordModifiedScopeProvider, target)
            if modification is KeywordContext.UNCHANGED:
                non_overwritten_targets.append(target)

        return targets
