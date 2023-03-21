from __future__ import annotations
import abc

import dataclasses
import typing

import libcst
from libcst import matchers as m, metadata as meta
from libcst.metadata.base_provider import (
    ProviderT,
)

from common.metadata.keyword_scopage import (
    KeywordContext,
    KeywordModifiedScopeProvider,
)

NAME = m.Name()
INSTANCE_ATTR = m.Attribute(m.Name("self"), m.Name())

TUPLE = m.Tuple()
LIST = m.List()

UNPACKABLE_ELEMENT = (m.StarredElement | m.Element)(NAME | INSTANCE_ATTR)


class Recognition:

    ## AnnAssign

    @staticmethod
    def instance_attribute_hint(
        metadata: typing.Mapping[ProviderT, typing.Mapping[libcst.CSTNode, object]],
        original_node: libcst.AnnAssign,
    ) -> Targets | None:
        """
        class Clazz:
            a: int
        """
        if original_node.value is None and _is_class_scope(metadata, original_node.target):
            return _access_targets(metadata, original_node.target)

        return None

    @staticmethod
    def libsa4py_hint(
        metadata: typing.Mapping[ProviderT, typing.Mapping[libcst.CSTNode, object]],
        original_node: libcst.Assign | libcst.AnnAssign,
    ) -> Targets | None:
        """
        class Clazz:
            a = ...
            a: int = ...
        """
        if m.matches(
            original_node, m.Assign(targets=[m.AssignTarget(NAME)], value=m.Ellipsis())
        ) and _is_class_scope(metadata, original_node.targets[0].target):
            return _access_targets(metadata, original_node.targets[0].target)

        elif m.matches(original_node, m.AnnAssign(value=m.Ellipsis())) and _is_class_scope(
            metadata, original_node.target
        ):
            return _access_targets(metadata, original_node.target)

        return None

    @staticmethod
    def annotated_hint(
        metadata: typing.Mapping[ProviderT, typing.Mapping[libcst.CSTNode, object]],
        original_node: libcst.AnnAssign,
    ) -> Targets | None:
        if (
            original_node.value is None
            or m.matches(original_node.value, m.Name("λ__LOWERED_HINT_MARKER__λ"))
        ) and not _is_class_scope(metadata, original_node.target):
            return _access_targets(metadata, original_node.target)

        return None

    @staticmethod
    def annotated_assignment(
        metadata: typing.Mapping[ProviderT, typing.Mapping[libcst.CSTNode, object]],
        original_node: libcst.AnnAssign,
    ) -> Targets | None:
        """
        class Clazz:
            a: int = 5
        
        a: int = 5

        r: requests.models.Response = ...
        """
        if (
            original_node.value is not None
            and not m.matches(original_node.value, m.Ellipsis())
            #and _is_class_scope(metadata, original_node.target)
        ):
            return _access_targets(metadata, original_node.target)

        # Support for stub files here
        elif m.matches(original_node.value, m.Ellipsis()) and not _is_class_scope(metadata, original_node.target):
            return _access_targets(metadata, original_node.target)

        return None

    @staticmethod
    def unannotated_assign_single_target(
        metadata: typing.Mapping[ProviderT, typing.Mapping[libcst.CSTNode, object]],
        original_node: libcst.Assign,
    ) -> Targets | None:
        """
        a = 10
        """
        if (
            len(original_node.targets) == 1
            and not m.matches(asstarget := original_node.targets[0], m.AssignTarget(LIST | TUPLE))
            and not m.matches(original_node.value, m.Ellipsis())
            # and not _is_class_scope(metadata, asstarget.target)
        ):
            return _access_targets(metadata, asstarget.target)
        return None

    @staticmethod
    def unannotated_assign_multiple_targets(
        metadata: typing.Mapping[ProviderT, typing.Mapping[libcst.CSTNode, object]],
        original_node: libcst.Assign,
    ) -> Targets | None:
        """
        class Clazz:
            a = b = ("Hello World",)

        a = b = 10
        a, b = 2, 10
        [a, (b, c)] = 10, (20, 30)
        """
        if (
            len(original_node.targets) > 1
            or m.matches(original_node.targets[0], m.AssignTarget(LIST | TUPLE))
        ): 
        #and not _is_class_scope(metadata, original_node.targets[0].target):
            unchanged, glbls, nonlocals = list(), list(), list()

            for discovered in (
                _access_targets(metadata, target.target) for target in original_node.targets
            ):
                unchanged.extend(discovered.unchanged)
                glbls.extend(discovered.glbls)
                nonlocals.extend(discovered.nonlocals)

            return Targets(unchanged, glbls, nonlocals)
        return None

    ## AugAssign

    @staticmethod
    def augassign_targets(
        metadata: typing.Mapping[ProviderT, typing.Mapping[libcst.CSTNode, object]],
        original_node: libcst.AugAssign,
    ) -> Targets:
        """
        a += 2
        """
        return _access_targets(metadata, original_node.target)

    ## For

    @staticmethod
    def for_targets(
        metadata: typing.Mapping[ProviderT, typing.Mapping[libcst.CSTNode, object]],
        original_node: libcst.For,
    ) -> Targets:
        """
        for x, y in zip(...):
            ...
        """
        return _access_targets(metadata, original_node.target)

    ## With
    @staticmethod
    def with_targets(
        metadata: typing.Mapping[ProviderT, typing.Mapping[libcst.CSTNode, object]],
        original_node: libcst.With,
    ) -> Targets:
        """
        with f() as g, h() as i:
            ...
        """
        unchanged, glbls, nonlocals = list(), list(), list()

        for targets in (
            _access_targets(metadata, item.asname.name)
            for item in original_node.items
            if item.asname is not None
        ):
            unchanged.extend(targets.unchanged)
            glbls.extend(targets.glbls)
            nonlocals.extend(targets.nonlocals)

        return Targets(unchanged, glbls, nonlocals)

    ## Misc

    @staticmethod
    def fallthru(original_node: libcst.CSTNode) -> typing.NoReturn:
        raise Exception(f"Cannot recognise {libcst.Module([]).code_for_node(original_node)}")


def _is_class_scope(
    metadata: typing.Mapping[ProviderT, typing.Mapping[libcst.CSTNode, object]],
    original_node: libcst.BaseAssignTargetExpression,
) -> bool:
    return isinstance(metadata[meta.ScopeProvider][original_node], meta.ClassScope)


@dataclasses.dataclass(frozen=True)
class Targets:
    unchanged: list[libcst.Name | libcst.Attribute] = dataclasses.field(default_factory=list)
    glbls: list[libcst.Name] = dataclasses.field(default_factory=list)
    nonlocals: list[libcst.Name] = dataclasses.field(default_factory=list)

    def empty(self):
        return not any((self.unchanged, self.glbls, self.nonlocals))


def _access_targets(
    metadata: typing.Mapping[ProviderT, typing.Mapping[libcst.CSTNode, object]],
    target: libcst.BaseAssignTargetExpression,
) -> Targets:
    if m.matches(target, NAME | INSTANCE_ATTR):
        targets = [target]

    elif m.matches(target, TUPLE | LIST):
        targets = [element.value for element in m.findall(target, UNPACKABLE_ELEMENT)]

    else:
        targets = []

    unchanged, glbls, nonlocals = list(), list(), list()
    for t in targets:
        match metadata[KeywordModifiedScopeProvider][t]:
            case KeywordContext.UNCHANGED:
                unchanged.append(t)
            case KeywordContext.GLOBAL:
                glbls.append(t)
            case KeywordContext.NONLOCAL:
                nonlocals.append(t)

    return Targets(
        unchanged=unchanged,
        glbls=glbls,
        nonlocals=nonlocals,
    )


class Matchers:
    annassign = m.AnnAssign(target=NAME | INSTANCE_ATTR)
    assign = m.Assign(
        targets=[m.AtLeastN(m.AssignTarget(target=NAME | INSTANCE_ATTR | LIST | TUPLE), n=1)]
    )
    augassign = m.AugAssign(target=NAME | INSTANCE_ATTR | TUPLE | LIST)
    fortargets = m.For(target=NAME | INSTANCE_ATTR | TUPLE | LIST)

    withitem = m.WithItem(asname=m.AsName(NAME | INSTANCE_ATTR | TUPLE | LIST))
    withitems = m.With(items=[m.AtLeastN(withitem, n=1)])


T = typing.TypeVar("T")


class Traverser(typing.Generic[T], abc.ABC):
    @abc.abstractmethod
    def instance_attribute_hint(self, original_node: libcst.AnnAssign, target: libcst.Name) -> T:
        """
        class C:
            a: int      # triggers
            a = ...     # ignored (libsa4py's instance attributes)
            a = 5       # ignored

        a: int          # ignored
        """
        ...

    @abc.abstractmethod
    def libsa4py_hint(self, original_node: libcst.Assign | libcst.AnnAssign, target: libcst.Name) -> T:
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
        original_node: libcst.AnnAssign,
        target: libcst.Name | libcst.Attribute,
    ) -> T:
        """
        a: int = 5      # triggers
        a: int          # ignored
        """
        ...

    @abc.abstractmethod
    def annotated_hint(
        self,
        original_node: libcst.AnnAssign,
        target: libcst.Name | libcst.Attribute,
    ) -> T:
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
        original_node: libcst.Assign,
        target: libcst.Name | libcst.Attribute,
    ) -> T:
        """
        class C:
            a: int      # ignored
            a = ...     # ignored
            a = 5       # triggers

        a = 10          # triggers
        a: int          # ignored

        a = b = 50          # ignored
        """
        ...

    @abc.abstractmethod
    def unannotated_assign_multiple_targets(
        self,
        original_node: libcst.Assign | libcst.AugAssign,
        target: libcst.Name | libcst.Attribute,
    ) -> T:
        """
        a = b = 50      # triggers
        """
        ...

    @abc.abstractmethod
    def for_target(self, original_node: libcst.For, target: libcst.Name | libcst.Attribute) -> T:
        """
        # triggers for both x and y
        for x, y in zip([1, 2, 3], "abc"):
            ...
        """
        ...

    @abc.abstractmethod
    def withitem_target(
        self,
        original_node: libcst.With,
        target: libcst.Name | libcst.Attribute,
    ) -> T:
        """
        # triggers for f
        with p.open() as f:
            ...
        """
        ...

    @abc.abstractmethod
    def global_target(
        self,
        original_node: libcst.Assign | libcst.AnnAssign | libcst.AugAssign,
        target: libcst.Name,
    ) -> T:
        ...

    @abc.abstractmethod
    def nonlocal_target(
        self,
        original_node: libcst.Assign | libcst.AnnAssign | libcst.AugAssign,
        target: libcst.Name,
    ) -> T:
        ...