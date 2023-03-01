import abc
import dataclasses
import itertools

import libcst
from libcst import metadata, matchers as m, helpers as h, codemod as c

from common.metadata import KeywordContext, KeywordModifiedScopeProvider
from .matchers import NAME, INSTANCE_ATTR, LIST, TUPLE


class ScopeAwareTransformer(m.MatcherDecoratableTransformer):
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
        self, function: libcst.FunctionDef
    ) -> libcst.BaseStatement | libcst.FlattenSentinel[
        libcst.BaseStatement
    ] | libcst.RemovalSentinel:
        if function.returns is not None:
            return self.annotated_function(function, function.returns)
        else:
            return self.unannotated_function(function)

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
    node: libcst.BaseSmallStatement


@dataclasses.dataclass
class Append:
    node: libcst.BaseSmallStatement


@dataclasses.dataclass
class Replace:
    old_node: libcst.CSTNode
    new_node: libcst.CSTNode


@dataclasses.dataclass
class Untouched:
    ...


Actions = list[Untouched | Prepend | Append | Replace]


def _handle_actions(
    updated_node: libcst.CSTNode, actions: Actions
) -> libcst.FlattenSentinel[libcst.BaseSmallStatement]:
    prepends: list[libcst.BaseSmallStatement] = []
    appends: list[libcst.BaseSmallStatement] = []

    for action in actions:
        match action:
            case Untouched():
                ...

            case Prepend(node):
                prepends.append(node)

            case Append(node):
                appends.append(node)

            case Replace(old_node, new_node):
                updated_node = updated_node.deep_replace(old_node, new_node)

    return libcst.FlattenSentinel(
        (
            *prepends,
            updated_node,
            *appends,
        )
    )


class HintableDeclarationTransformer(c.ContextAwareTransformer, abc.ABC):
    """
    Provide hook methods for transforming hintable attributes (both a and self.a)
    in Assign, AnnAssign and AugAssign, as well as WithItems, For Loops and Walrus usages.
    """

    METADATA_DEPENDENCIES = (
        KeywordModifiedScopeProvider,
        metadata.ParentNodeProvider,
        metadata.ScopeProvider,
    )

    @abc.abstractmethod
    def instance_attribute_hint(
        self, assignment: libcst.AnnAssign, target: libcst.Name, annotation: libcst.Annotation
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
    def libsa4py_hint(self, assignment: libcst.Assign, target: libcst.Name) -> Actions:
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
    ) -> Actions:
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
        assign: libcst.Assign | libcst.AugAssign,
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
        assign: libcst.Assign | libcst.AugAssign,
        target: libcst.Name | libcst.Attribute,
    ) -> Actions:
        """
        a = b = 50      # triggers
        """
        ...

    @abc.abstractmethod
    def for_target(self, forloop: libcst.For, target: libcst.Name | libcst.Attribute) -> Actions:
        ...

    @abc.abstractmethod
    def compfor_target(
        self, forloop: libcst.CompFor, target: libcst.Name | libcst.Attribute
    ) -> Actions:
        ...

    @abc.abstractmethod
    def withitem_target(
        self,
        with_node: libcst.With,
        withitem: libcst.WithItem,
        target: libcst.Name | libcst.Attribute,
    ) -> Actions:
        ...

    @abc.abstractmethod
    def global_target(
        self, assign: libcst.Assign | libcst.AnnAssign | libcst.AugAssign, target: libcst.Name
    ) -> Actions:
        ...

    @abc.abstractmethod
    def nonlocal_target(
        self, assign: libcst.Assign | libcst.AnnAssign | libcst.AugAssign, target: libcst.Name
    ) -> Actions:
        ...

    @m.call_if_inside(m.AnnAssign(target=NAME | INSTANCE_ATTR))
    def leave_AnnAssign(
        self, _: libcst.AnnAssign, updated_node: libcst.AnnAssign
    ) -> libcst.FlattenSentinel[libcst.BaseSmallStatement]:
        if updated_node.value is not None:
            actions = self.annotated_assignment(
                updated_node, updated_node.target, updated_node.annotation
            )
        elif isinstance(
            self.get_metadata(metadata.ScopeProvider, updated_node), metadata.ClassScope
        ):
            actions = self.instance_attribute_hint(
                updated_node, updated_node.target, updated_node.annotation
            )
        else:
            actions = self.annotated_hint(
                updated_node, updated_node.target, updated_node.annotation
            )

        return _handle_actions(updated_node, actions)

    @m.call_if_inside(
        m.Assign(
            targets=[m.AtLeastN(m.AssignTarget(target=NAME | INSTANCE_ATTR | LIST | TUPLE), n=1)]
        )
    )
    def leave_Assign(
        self, _: libcst.Assign, updated_node: libcst.Assign
    ) -> libcst.FlattenSentinel[libcst.BaseSmallStatement]:
        # Catch libsa4py's retainment of INSTANCE_ATTRs
        if m.matches(
            updated_node, m.Assign(targets=[m.AssignTarget(target=NAME)], value=m.Ellipsis())
        ) and isinstance(
            self.get_metadata(metadata.ScopeProvider, updated_node.targets[0].target),
            metadata.ClassScope,
        ):
            actions = self.libsa4py_hint(updated_node, updated_node.targets[0].target)

        elif len(updated_node.targets) == 1 and not m.matches(
            asstarget := updated_node.targets[0], m.AssignTarget(LIST | TUPLE)
        ):
            targets_of_interest = self._access_targets(asstarget.target)
            assert len(targets_of_interest) <= 1
            if targets_of_interest:
                actions = self.unannotated_assign_single_target(
                    updated_node, targets_of_interest[0]
                )
            else:
                actions = Actions()

        else:
            targets_of_interest = list(
                itertools.chain.from_iterable(
                    self._access_targets(asstarget.target) for asstarget in updated_node.targets
                )
            )
            actions = list(
                itertools.chain.from_iterable(
                    self.unannotated_assign_multiple_targets(updated_node, target)
                    for target in targets_of_interest
                )
            )

        return _handle_actions(updated_node, actions)

    @m.call_if_inside(m.AugAssign(target=NAME | INSTANCE_ATTR | TUPLE | LIST))
    def leave_AugAssign(
        self, _: libcst.AugAssign, updated_node: libcst.AugAssign
    ) -> libcst.FlattenSentinel[libcst.BaseSmallStatement]:
        targets_of_interest = self._access_targets(updated_node.target)
        actions = list(
            itertools.chain.from_iterable(
                self.unannotated_assign_multiple_targets(updated_node, target)
                for target in targets_of_interest
            )
        )

        return _handle_actions(updated_node, actions)

    # @abc.abstractmethod
    # def namedexpr_target(
    #     self, walrus: libcst.NamedExpr, target: libcst.Name | libcst.Attribute
    # ) -> Actions:
    #     ...

    @m.call_if_inside(m.For(target=NAME | INSTANCE_ATTR | TUPLE | LIST))
    def leave_For(
        self, _: libcst.For, updated_node: libcst.For
    ) -> libcst.FlattenSentinel[libcst.BaseSmallStatement]:
        targets_of_interest = self._access_targets(updated_node.target)
        actions = list(
            itertools.chain.from_iterable(
                self.for_target(updated_node, target) for target in targets_of_interest
            )
        )

        return _handle_actions(updated_node, actions)

    @m.call_if_inside(m.CompFor(target=NAME | INSTANCE_ATTR | TUPLE | LIST))
    def leave_CompFor(
        self, _: libcst.CompFor, updated_node: libcst.CompFor
    ) -> libcst.FlattenSentinel[libcst.CSTNode]:
        targets_of_interest = self._access_targets(updated_node.target)
        actions = list(
            itertools.chain.from_iterable(
                self.compfor_target(updated_node, target) for target in targets_of_interest
            )
        )

        return _handle_actions(updated_node, actions)

    @m.call_if_inside(m.WithItem(asname=m.AsName(NAME | INSTANCE_ATTR | TUPLE | LIST)))
    def leave_WithItem(self, updated_node: libcst.WithItem) -> None:
        with_node = self.get_metadata(metadata.ParentNodeProvider, updated_node)
        targets_of_interest = self._access_targets(updated_node.asname.name)
        actions = list(
            itertools.chain.from_iterable(
                self.withitem_target(with_node, updated_node, target)
                for target in targets_of_interest
            )
        )

        return _handle_actions(updated_node, actions)

    # TODO: Can the prependable / appendable node for NamedExpr be found by
    # TODO: using m.StatementLine(m.AtLeastN(m.NamedExpr(...), n=1))
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

        targets = list(
            filter(
                lambda t: self.get_metadata(KeywordModifiedScopeProvider, t)
                is KeywordContext.UNCHANGED,
                targets,
            )
        )

        return targets
