import abc
import dataclasses
import itertools
import typing
from typing import Union, Optional

import libcst
from libcst import metadata, matchers as m, helpers as h, codemod as c

from common.metadata.keyword_scopage import KeywordModifiedScopeProvider

from . import _traversal


class ScopeAwareTransformer(c.ContextAwareTransformer):
    def __init__(self, context: c.CodemodContext) -> None:
        super().__init__(context)
        self._qualifier: list[str] = []

    def qualified_scope(self) -> tuple[str, ...]:
        return tuple(self._qualifier)

    def qualified_name(self, name: Union[libcst.CSTNode, str]) -> str:
        name = h.get_full_name_for_node_or_raise(name)
        return ".".join((*self._qualifier, name))

    @m.visit(m.FunctionDef() | m.ClassDef())
    def __on_enter_scope(self, node: Union[libcst.FunctionDef, libcst.ClassDef]) -> None:
        self._qualifier.append(node.name.value)

    @m.leave(m.FunctionDef() | m.ClassDef())
    def __on_leave_scope(
        self,
        _1: Union[libcst.FunctionDef, libcst.ClassDef],
        _2: Union[libcst.FunctionDef, libcst.ClassDef],
    ) -> Union[libcst.FunctionDef, libcst.ClassDef]:
        self._qualifier.pop()
        return _2


class HintableParameterTransformer(c.ContextAwareTransformer, abc.ABC):
    def leave_Param(
        self, original_node: libcst.Param, updated_node: libcst.Param
    ) -> Union[libcst.Param, libcst.MaybeSentinel, libcst.FlattenSentinel[
        libcst.Param
    ], libcst.RemovalSentinel]:
        if updated_node.annotation is not None:
            return self.annotated_param(updated_node, updated_node.annotation)
        else:
            return self.unannotated_param(updated_node)

    @abc.abstractmethod
    def annotated_param(
        self, param: libcst.Param, annotation: libcst.Annotation
    ) -> Union[libcst.Param, libcst.MaybeSentinel, libcst.FlattenSentinel[
        libcst.Param
    ], libcst.RemovalSentinel]:
        ...

    @abc.abstractmethod
    def unannotated_param(
        self, param: libcst.Param
    ) -> Union[libcst.Param, libcst.MaybeSentinel, libcst.FlattenSentinel[
        libcst.Param
    ], libcst.RemovalSentinel]:
        ...


class HintableReturnTransformer(c.ContextAwareTransformer, abc.ABC):
    def leave_FunctionDef(
        self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef
    ) -> Union[libcst.BaseStatement, libcst.FlattenSentinel[
        libcst.BaseStatement
    ], libcst.RemovalSentinel]:
        if updated_node.returns is not None:
            return self.annotated_function(updated_node, updated_node.returns)
        else:
            return self.unannotated_function(updated_node)

    @abc.abstractmethod
    def annotated_function(
        self, function: libcst.FunctionDef, annotation: libcst.Annotation
    ) -> Union[libcst.BaseStatement, libcst.FlattenSentinel[
        libcst.BaseStatement
    ], libcst.RemovalSentinel]:
        ...

    @abc.abstractmethod
    def unannotated_function(
        self, function: libcst.FunctionDef
    ) -> Union[libcst.BaseStatement, libcst.FlattenSentinel[
        libcst.BaseStatement
    ], libcst.RemovalSentinel]:
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
class Remove:
    ...


@dataclasses.dataclass(frozen=True)
class Untouched:
    ...


Actions = list[Union[Untouched, Prepend, Append, Replace, Remove]]


class HintableDeclarationTransformer(
    c.ContextAwareTransformer, _traversal.Traverser[Actions], abc.ABC
):
    """
    Provide hook methods for transforming hintable attributes (both a and self.a)
    in Assign, AnnAssign and AugAssign, as well as WithItems, For Loops
    """

    METADATA_DEPENDENCIES = (
        metadata.ParentNodeProvider,
        metadata.ScopeProvider,
        KeywordModifiedScopeProvider,
    )

    @m.call_if_inside(_traversal.Matchers.annassign)
    def leave_AnnAssign(
        self, original_node: libcst.AnnAssign, updated_node: libcst.AnnAssign
    ) -> Union[libcst.FlattenSentinel[
        libcst.BaseSmallStatement
    ], libcst.RemovalSentinel, libcst.BaseSmallStatement]:
        if targets := _traversal.Recognition.instance_attribute_hint(self.metadata, original_node):
            transformer = self.instance_attribute_hint

        elif targets := _traversal.Recognition.libsa4py_hint(self.metadata, original_node):
            transformer = self.libsa4py_hint

        elif targets := _traversal.Recognition.annotated_hint(self.metadata, original_node):
            transformer = self.annotated_hint

        elif targets := _traversal.Recognition.annotated_assignment(self.metadata, original_node):
            transformer = self.annotated_assignment

        else:
            _traversal.Recognition.fallthru(original_node)
            return original_node

        return self._apply_actions(targets, transformer, original_node, updated_node)

    @m.call_if_inside(_traversal.Matchers.assign)
    def leave_Assign(
        self, original_node: libcst.Assign, updated_node: libcst.Assign
    ) -> Union[libcst.FlattenSentinel[
        libcst.BaseSmallStatement
    ], libcst.RemovalSentinel, libcst.BaseSmallStatement]:
        if targets := _traversal.Recognition.libsa4py_hint(self.metadata, original_node):
            transformer = self.libsa4py_hint

        elif targets := _traversal.Recognition.unannotated_assign_single_target(
            self.metadata, original_node
        ):
            transformer = self.unannotated_assign_single_target

        elif targets := _traversal.Recognition.unannotated_assign_multiple_targets(
            self.metadata, original_node
        ):
            transformer = self.unannotated_assign_multiple_targets_or_augassign

        else:
            _traversal.Recognition.fallthru(original_node)
            return original_node

        return self._apply_actions(targets, transformer, original_node, updated_node)

    @m.call_if_inside(_traversal.Matchers.augassign)
    def leave_AugAssign(
        self, original_node: libcst.AugAssign, updated_node: libcst.AugAssign
    ) -> Union[libcst.FlattenSentinel[
        libcst.BaseSmallStatement
    ], libcst.RemovalSentinel, libcst.BaseSmallStatement]:
        if targets := _traversal.Recognition.augassign_targets(self.metadata, original_node):
            transformer = self.unannotated_assign_multiple_targets_or_augassign
        else:
            _traversal.Recognition.fallthru(original_node)
            return original_node

        return self._apply_actions(targets, transformer, original_node, updated_node)

    @m.call_if_inside(_traversal.Matchers.fortargets)
    def leave_For(
        self, original_node: libcst.For, updated_node: libcst.For
    ) -> Union[libcst.FlattenSentinel[libcst.BaseStatement], libcst.RemovalSentinel, libcst.BaseStatement]:
        if targets := _traversal.Recognition.for_targets(self.metadata, original_node):
            transformer = self.for_target
        else:
            _traversal.Recognition.fallthru(original_node)
            return original_node

        return self._apply_actions(targets, transformer, original_node, updated_node)

    @m.call_if_inside(_traversal.Matchers.withitems)
    def leave_With(
        self, original_node: libcst.With, updated_node: libcst.With
    ) -> Union[libcst.FlattenSentinel[
        libcst.BaseStatement
    ], libcst.RemovalSentinel, libcst.BaseStatement]:
        if targets := _traversal.Recognition.with_targets(self.metadata, original_node):
            transformer = self.withitem_target
        else:
            _traversal.Recognition.fallthru(original_node)
            return original_node

        return self._apply_actions(targets, transformer, original_node, updated_node)

    # Visitors ignore Lambdas, so Transformer should too
    def visit_Lambda(self, node: libcst.Lambda) -> Optional[bool]:
        return False

    _T = typing.TypeVar("_T", libcst.BaseSmallStatement, libcst.BaseStatement)

    def _apply_actions(
        self,
        targets: _traversal.Targets,
        transformer: typing.Callable[[_T, Union[libcst.Name, libcst.Attribute]], Actions],
        original_node: _T,
        updated_node: _T,
    ) -> Union[libcst.FlattenSentinel[_T], libcst.RemovalSentinel]:

        unchanged_actions = list(
            itertools.chain.from_iterable(
                transformer(original_node, target) for target in targets.unchanged
            )
        )
        global_actions = list(
            itertools.chain.from_iterable(
                self.global_target(original_node, target) for target in targets.glbls
            )
        )
        nonlocal_actions = list(
            itertools.chain.from_iterable(
                self.nonlocal_target(original_node, target) for target in targets.nonlocals
            )
        )

        prepends = []
        appends = []

        for action in itertools.chain(unchanged_actions, global_actions, nonlocal_actions):
            if isinstance(action, Untouched):
                ...

            elif isinstance(action, Prepend):
                prepends.append(action.node)

            elif isinstance(action, Append):
                appends.append(action.node)

            elif isinstance(action, Replace):
                updated_node = m.replace(updated_node, action.matcher, action.replacement)

            elif isinstance(action, Remove):
                updated_node = libcst.RemoveFromParent()
                return updated_node

        if isinstance(updated_node, libcst.BaseSmallStatement):
            # Must return libcst.FlattenSentinel[BaseSmallStatement]
            return libcst.FlattenSentinel(
                (
                    *prepends,
                    # libcst.EmptyLine(),
                    updated_node,
                    # libcst.EmptyLine(),
                    *appends,
                )
            )
        else:
            # Must return libcst.FlattenSentinel[BaseStatement]
            return libcst.FlattenSentinel(
                (
                    libcst.SimpleStatementLine(body=prepends),
                    updated_node,
                    libcst.SimpleStatementLine(body=appends),
                )
            )
