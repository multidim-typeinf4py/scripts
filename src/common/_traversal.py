from __future__ import annotations

import abc
import dataclasses
import typing

import libcst
from libcst import matchers as m, metadata as meta

from src.common.metadata.keyword_scopage import (
    KeywordContext,
    KeywordModifiedScopeProvider,
)

NAME = m.Name()
INSTANCE_ATTR = m.Attribute(m.Name("self"), m.Name())

TUPLE = m.Tuple()
LIST = m.List()

UNPACKABLE_ELEMENT = (m.StarredElement | m.Element)(NAME | INSTANCE_ATTR)


class Recognition(libcst.MetadataDependent):
    ## AnnAssign

    def extract_instance_attribute_hint(
        self,
        original_node: libcst.AnnAssign,
    ) -> Targets | None:
        """
        class Clazz:
            a: int
        """
        if original_node.value is None and self._is_class_scope(original_node.target):
            return Targets.from_accesses(self.metadata, original_node.target)

        return None

    def extract_libsa4py_hint(
        self,
        original_node: typing.Union[libcst.AnnAssign, libcst.Assign],
    ) -> Targets | None:
        """
        class Clazz:
            a = ...
            a: int = ...
        """
        if m.matches(
            original_node, m.Assign(targets=[m.AssignTarget(NAME)], value=m.Ellipsis())
        ) and self._is_class_scope(original_node.targets[0].target):
            return Targets.from_accesses(self.metadata, original_node.targets[0].target)

        elif m.matches(
            original_node, m.AnnAssign(value=m.Ellipsis())
        ) and self._is_class_scope(original_node.target):
            return Targets.from_accesses(self.metadata, original_node.target)

        return None

    def extract_annotated_hint(
        self,
        original_node: libcst.AnnAssign,
    ) -> Targets | None:
        if (
            original_node.value is None
            or m.matches(original_node.value, m.Name("λ__LOWERED_HINT_MARKER__λ"))
        ) and not self._is_class_scope(original_node.target):
            return Targets.from_accesses(self.metadata, original_node.target)

        return None

    def extract_annotated_assignment(
        self,
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
            # and _is_class_scope(self.metadata, original_node.target)
        ):
            return Targets.from_accesses(self.metadata, original_node.target)

        # Support for stub files here
        elif m.matches(original_node.value, m.Ellipsis()) and not self._is_class_scope(
            original_node.target
        ):
            return Targets.from_accesses(self.metadata, original_node.target)

        return None

    def extract_unannotated_assign_single_target(
        self,
        original_node: libcst.Assign,
    ) -> Targets | None:
        """
        a = 10
        """
        if (
            len(original_node.targets) == 1
            and not m.matches(
                asstarget := original_node.targets[0], m.AssignTarget(LIST | TUPLE)
            )
            and not m.matches(original_node.value, m.Ellipsis())
            # and not _is_class_scope(self.metadata, asstarget.target)
        ):
            return Targets.from_accesses(self.metadata, asstarget.target)
        return None

    def extract_unannotated_assign_multiple_targets(
        self,
        original_node: libcst.Assign,
    ) -> Targets | None:
        """
        class Clazz:
            a = b = ("Hello World",)

        a = b = 10
        a, b = 2, 10
        [a, (b, c)] = 10, (20, 30)
        """
        if len(original_node.targets) > 1 or m.matches(
            original_node.targets[0], m.AssignTarget(LIST | TUPLE)
        ):
            unchanged, glbls, nonlocals = list(), list(), list()

            for discovered in (
                Targets.from_accesses(self.metadata, target.target)
                for target in original_node.targets
            ):
                unchanged.extend(discovered.unchanged)
                glbls.extend(discovered.glbls)
                nonlocals.extend(discovered.nonlocals)

            return Targets(unchanged, glbls, nonlocals)
        return None

    ## AugAssign

    def extract_augassign(
        self,
        original_node: libcst.AugAssign,
    ) -> Targets:
        """
        a += 2
        """
        return Targets.from_accesses(self.metadata, original_node.target)

    ## For

    def extract_for(
        self,
        original_node: libcst.For,
    ) -> Targets:
        """
        for x, y in zip(...):
            ...
        """
        return Targets.from_accesses(self.metadata, original_node.target)

    ## With

    def with_targets(
        self,
        original_node: libcst.With,
    ) -> Targets:
        """
        with f() as g, h() as i:
            ...
        """
        unchanged, glbls, nonlocals = list(), list(), list()

        for targets in (
            Targets.from_accesses(self.metadata, item.asname.name)
            for item in original_node.items
            if item.asname is not None
        ):
            unchanged.extend(targets.unchanged)
            glbls.extend(targets.glbls)
            nonlocals.extend(targets.nonlocals)

        return Targets(unchanged, glbls, nonlocals)

    ## Misc

    def fallthru(self, original_node: libcst.CSTNode) -> Targets:
        print(
            f"WARNING: Cannot recognise {libcst.Module([]).code_for_node(original_node)}"
        )
        return Targets(list(), list(), list())

    def _is_class_scope(self, original_node: libcst.BaseAssignTargetExpression) -> bool:
        scope = self.get_metadata(meta.ScopeProvider, original_node)
        return isinstance(scope, meta.ClassScope)


@dataclasses.dataclass(frozen=True)
class Targets:
    unchanged: list[typing.Union[libcst.Name, libcst.Attribute]] = dataclasses.field(
        default_factory=list
    )
    glbls: list[libcst.Name] = dataclasses.field(default_factory=list)
    nonlocals: list[libcst.Name] = dataclasses.field(default_factory=list)

    def empty(self):
        return not any((self.unchanged, self.glbls, self.nonlocals))

    @staticmethod
    def from_accesses(
        metadata,
        target: libcst.BaseAssignTargetExpression,
    ) -> Targets:
        if m.matches(target, NAME | INSTANCE_ATTR):
            targets = [target]

        elif m.matches(target, TUPLE | LIST):
            # Explicitly go down tree one level at a time, be careful not to extract from subscript or similar
            candidates: list[libcst.Tuple | libcst.List] = [target]
            targets = []

            while intermediaries := [
                element
                for cand in candidates
                for element in cand.elements
                if m.matches(
                    element,
                    (m.StarredElement | m.Element)(NAME | INSTANCE_ATTR | LIST | TUPLE),
                )
            ]:
                targets.extend(
                    element.value
                    for element in intermediaries
                    if m.matches(
                        element, (m.StarredElement | m.Element)(NAME | INSTANCE_ATTR)
                    )
                )
                candidates = [
                    element.value
                    for element in intermediaries
                    if m.matches(element, (m.StarredElement | m.Element)(LIST | TUPLE))
                ]

        else:
            targets = []

        unchanged, glbls, nonlocals = list(), list(), list()
        for t in targets:
            scopage = metadata[KeywordModifiedScopeProvider][t]
            if scopage is KeywordContext.UNCHANGED:
                unchanged.append(t)
            elif scopage is KeywordContext.GLOBAL:
                glbls.append(t)
            elif scopage is KeywordContext.NONLOCAL:
                nonlocals.append(t)

        return Targets(
            unchanged=unchanged,
            glbls=glbls,
            nonlocals=nonlocals,
        )


class Matchers:
    annassign = m.AnnAssign(target=NAME | INSTANCE_ATTR)
    assign = m.Assign(
        targets=[
            m.AtLeastN(m.AssignTarget(target=NAME | INSTANCE_ATTR | LIST | TUPLE), n=1)
        ]
    )
    augassign = m.AugAssign(target=NAME | INSTANCE_ATTR | TUPLE | LIST)
    fortargets = m.For(target=NAME | INSTANCE_ATTR | TUPLE | LIST)

    withitem = m.WithItem(asname=m.AsName(NAME | INSTANCE_ATTR | TUPLE | LIST))
    withitems = m.With(items=[m.AtLeastN(withitem, n=1)])


T = typing.TypeVar("T")


class Traverser(typing.Generic[T], abc.ABC):
    @abc.abstractmethod
    def instance_attribute_hint(
        self, original_node: libcst.AnnAssign, target: libcst.Name
    ) -> T:
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
        self,
        original_node: typing.Union[libcst.Name, libcst.AnnAssign],
        target: libcst.Name,
    ) -> T:
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
        target: typing.Union[libcst.Name, libcst.Attribute],
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
        target: typing.Union[libcst.Name, libcst.Attribute],
    ) -> T:
        """
        a: int = 5      # ignored
        a: int          # triggers

        class C:
            a: int      # ignored
        """
        ...

    @abc.abstractmethod
    def assign_single_target(
        self,
        original_node: libcst.Assign,
        target: typing.Union[libcst.Name, libcst.Attribute],
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
    def assign_multiple_targets_or_augassign(
        self,
        original_node: typing.Union[libcst.Assign, libcst.AugAssign],
        target: typing.Union[libcst.Name, libcst.Attribute],
    ) -> T:
        """
        a = b = 50      # triggers
        """
        ...

    @abc.abstractmethod
    def for_target(
        self,
        original_node: libcst.For,
        target: typing.Union[libcst.Name, libcst.Attribute],
    ) -> T:
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
        target: typing.Union[libcst.Name, libcst.Attribute],
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
        original_node: typing.Union[libcst.Name, libcst.AnnAssign] | libcst.AugAssign,
        target: libcst.Name,
    ) -> T:
        ...

    @abc.abstractmethod
    def nonlocal_target(
        self,
        original_node: typing.Union[libcst.Name, libcst.AnnAssign] | libcst.AugAssign,
        target: libcst.Name,
    ) -> T:
        ...
