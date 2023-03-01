from __future__ import annotations
import collections
import enum

import libcst
from libcst import metadata, matchers as m


from .matchers import NAME


class KeywordContext(enum.Enum):
    GLOBAL = enum.auto()
    NONLOCAL = enum.auto()
    UNCHANGED = enum.auto()


class _KeywordModifiedScopeVisitor(m.MatcherDecoratableVisitor):
    def __init__(self, provider: KeywordModifiedScopeProvider) -> None:
        super().__init__()
        self.provider = provider

        self.active_scopes: list[metadata.FunctionScope] = []

        self._scope2nonlocal: collections.defaultdict[
            metadata.Scope, set[str]
        ] = collections.defaultdict(set)
        self._scope2global: collections.defaultdict[
            metadata.Scope, set[str]
        ] = collections.defaultdict(set)

    def visit_Global_names(self, node: libcst.Global) -> bool | None:
        self._scope2global[self.active_scopes[-1]] |= set(
            nameitem.name.value for nameitem in node.names
        )

    def visit_Nonlocal_names(self, node: libcst.Nonlocal) -> None:
        self._scope2nonlocal[self.active_scopes[-1]] |= set(
            nameitem.name.value for nameitem in node.names
        )

    @m.call_if_inside(
        m.AssignTarget(target=NAME) | m.AugAssign(target=NAME) | m.AnnAssign(target=NAME)
    )
    def visit_Name(self, node: libcst.Name) -> bool | None:
        if not self.active_scopes or isinstance(self.active_scopes[-1], metadata.ClassScope):
            self.provider.set_metadata(node, KeywordContext.UNCHANGED)
        elif node.value in self._scope2nonlocal.get(self.active_scopes[-1], set()):
            self.provider.set_metadata(node, KeywordContext.NONLOCAL)
        elif node.value in self._scope2global.get(self.active_scopes[-1], set()):
            self.provider.set_metadata(node, KeywordContext.GLOBAL)
        else:
            self.provider.set_metadata(node, KeywordContext.UNCHANGED)

    def visit_FunctionDef_body(self, node: libcst.FunctionDef) -> None:
        self.active_scopes.append(self.provider.get_metadata(metadata.ScopeProvider, node.body.body[0]))

    def leave_FunctionDef_body(self, _: libcst.FunctionDef) -> None:
        self._scope2nonlocal.pop(self.active_scopes[-1], None)
        self._scope2global.pop(self.active_scopes[-1], None)

        self.active_scopes.pop()

    def visit_ClassDef_body(self, node: libcst.ClassDef) -> None:
        self.active_scopes.append(self.provider.get_metadata(metadata.ScopeProvider, node.body.body[0]))

    def leave_ClassDef_body(self, _: libcst.ClassDef) -> None:
        self._scope2nonlocal.pop(self.active_scopes[-1], None)
        self._scope2global.pop(self.active_scopes[-1], None)

        self.active_scopes.pop()


class KeywordModifiedScopeProvider(metadata.BatchableMetadataProvider[KeywordContext]):
    METADATA_DEPENDENCIES = (metadata.ScopeProvider,)

    def visit_Module(self, node: libcst.Module) -> bool | None:
        node.visit(_KeywordModifiedScopeVisitor(self))
