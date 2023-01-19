import builtins
import collections
import pathlib

import pandera.typing as pt

import pandas as pd
import libcst as cst
import libcst.metadata as metadata

from common.schemas import (
    ContextCategory,
    ContextSymbolSchema,
    ContextSymbolSchemaColumns,
    TypeCollectionCategory,
)
from common._helper import _stringify

from context.features import RelevantFeatures


def generate_context_vectors_for_file(
    features: RelevantFeatures, repo: pathlib.Path, path: pathlib.Path
) -> pt.DataFrame[ContextSymbolSchema] | None:
    visitor = ContextVectorVisitor(filepath=str(path.relative_to(repo)), features=features)
    module = cst.parse_module(path.open().read())

    md = metadata.MetadataWrapper(module)
    md.visit(visitor)

    return visitor.build()


class ContextVectorVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (metadata.ScopeProvider,)

    def __init__(self, filepath: str, features: RelevantFeatures) -> None:
        self.features: RelevantFeatures = features

        self.scope_stack: list[tuple[str, ...]] = []
        self.filepath = filepath
        self.loop_stack: list[cst.CSTNode] = []

        self.dfrs: list[tuple[str, str, str, int, int, int, int, int]] = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        self._handle_annotatable(
            annotatable=node,
            identifier=node.name.value,
            annotation=node.returns,
            category=TypeCollectionCategory.CALLABLE_RETURN,
        )

        self._update_scope_stack(node.name.value)

    def visit_Param(self, node: cst.Param) -> bool | None:
        self._handle_annotatable(
            annotatable=node,
            identifier=node.name.value,
            annotation=node.annotation,
            category=TypeCollectionCategory.CALLABLE_PARAMETER,
        )

    def leave_FunctionDef(self, _: "cst.FunctionDef") -> None:
        self._leave_scope()

    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
        self._update_scope_stack(node.name.value)

    def leave_ClassDef(self, _: cst.ClassDef) -> None:
        self._leave_scope()

    def visit_AssignTarget(self, node: cst.AssignTarget) -> bool | None:
        ident = _stringify(node.target)
        self._handle_annotatable(
            annotatable=node.target,
            identifier=ident,
            annotation=None,
            category=TypeCollectionCategory.VARIABLE,
        )

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool | None:
        match self.get_metadata(metadata.ScopeProvider, node):
            case metadata.ClassScope():
                category = TypeCollectionCategory.CLASS_ATTR

            case _:
                category = TypeCollectionCategory.VARIABLE

        ident = _stringify(node.target)
        self._handle_annotatable(
            annotatable=node.target,
            identifier=ident,
            annotation=node.annotation,
            category=category,
        )

    def _update_scope_stack(self, name: str) -> None:
        self.scope_stack.append((*self.scope(), name))

    def _leave_scope(self) -> None:
        self.scope_stack.pop()

    def scope(self) -> tuple[str, ...]:
        return self.scope_stack[-1] if self.scope_stack else tuple()

    def qname_within_scope(self, identifier: str) -> str:
        if s := self.scope():
            return f"{'.'.join(s)}.{identifier}"
        return identifier

    def visit_While(self, node: cst.While) -> bool | None:
        self.loop_stack.append(node)

    def leave_While(self, _: cst.While) -> None:
        self.loop_stack.pop()

    def visit_For(self, node: cst.For) -> bool | None:
        self.loop_stack.append(node)

    def leave_For(self, _: cst.For) -> None:
        self.loop_stack.pop()

    def _handle_annotatable(
        self,
        annotatable: cst.CSTNode,
        identifier: str,
        annotation: cst.Annotation | None,
        category: TypeCollectionCategory,
    ) -> None:
        loopf = int(self.features.loop and self._is_in_loop())
        reassignedf = int(self.features.reassigned and self._is_reassigned(annotatable))
        nestedf = int(self.features.nested and self._is_nested_scope(annotatable))
        user_definedf = int(self.features.user_defined and self._is_userdefined(annotation))

        categoryf = self._ctxt_category(annotatable, category)

        self.dfrs.append(
            (
                self.filepath,
                category,
                self.qname_within_scope(identifier),
                loopf,
                reassignedf,
                nestedf,
                user_definedf,
                categoryf,
            )
        )

    def _is_in_loop(self) -> bool:
        return bool(self.loop_stack)

    def _is_reassigned(self, node: cst.CSTNode) -> bool:
        scope: metadata.Scope = self.get_metadata(metadata.ScopeProvider, node)
        scope_assgns = scope.assignments[node]

        return len(scope_assgns) >= 2

    def _is_nested_scope(self, node: cst.CSTNode) -> bool:
        # Detect class in class or function in function
        scopes = [scope := self.get_metadata(metadata.ScopeProvider, node)]
        while not (scope is None or isinstance(scope, metadata.BuiltinScope)):
            scopes.append(scope := scope.parent)

        counted = collections.Counter(map(type, scopes))

        fncount = counted.get(metadata.FunctionScope, 0) + isinstance(node, cst.FunctionDef)
        czcount = counted.get(metadata.ClassScope, 0) + isinstance(node, metadata.ClassScope)
        return fncount >= 2 or czcount >= 2

    def _is_userdefined(self, annotation: cst.Annotation | None) -> bool:
        if annotation is None:
            return False

        a = _stringify(annotation.annotation)
        sanitised = "".join(a.split())
        unions = sanitised.split("|")

        return any(u not in dir(builtins) for u in unions)

    def _ctxt_category(
        self, annotatable: cst.CSTNode, category: TypeCollectionCategory
    ) -> ContextCategory:
        match category:
            case TypeCollectionCategory.CALLABLE_RETURN:
                return ContextCategory.CALLABLE_RETURN

            case TypeCollectionCategory.CALLABLE_PARAMETER:
                return ContextCategory.CALLABLE_PARAMETER

            case TypeCollectionCategory.CLASS_ATTR:
                return ContextCategory.CLASS_ATTR

            case TypeCollectionCategory.VARIABLE:
                return (
                    ContextCategory.VARIABLE
                    if isinstance(annotatable, cst.Name)
                    else ContextCategory.INSTANCE_ATTR
                )


    def build(self) -> pt.DataFrame[ContextSymbolSchema] | None:
        if not self.dfrs:
            return None

        df = pt.DataFrame[ContextSymbolSchema](self.dfrs, columns=ContextSymbolSchemaColumns)

        # Reassigned variables also mean redefined parameters
        variables = df[
            (df[ContextSymbolSchema.category] == TypeCollectionCategory.VARIABLE)
            & (df[ContextSymbolSchema.reassigned] == 1)
        ]
        # Find parameters by the same name
        parameters = df[ContextSymbolSchema.qname].isin(variables[ContextSymbolSchema.qname])
        df.loc[parameters, ContextSymbolSchema.reassigned] = 1
        return df