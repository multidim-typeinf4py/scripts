import builtins
import collections
import pathlib

import pandera.typing as pt

import pandas as pd
from pandas._libs import missing

import libcst as cst
import libcst.metadata as metadata
from libcst import matchers as m
from libcst.helpers import get_full_name_for_node_or_raise

from common.schemas import (
    ContextCategory,
    ContextSymbolSchema,
    ContextSymbolSchemaColumns,
    TypeCollectionCategory,
)
from common.ast_helper import _stringify, generate_qname_ssas_for_file

from context.features import RelevantFeatures


def generate_context_vectors_for_file(
    features: RelevantFeatures, repo: pathlib.Path, path: pathlib.Path
) -> pt.DataFrame[ContextSymbolSchema]:
    visitor = ContextVectorVisitor(filepath=str(path.relative_to(repo)), features=features)
    module = cst.parse_module(path.open().read())

    md = metadata.MetadataWrapper(module)
    md.visit(visitor)

    return visitor.build()


class ContextVectorVisitor(m.MatcherDecoratableVisitor):
    ContextVector = collections.namedtuple(
        "ContextVector",
        [
            ContextSymbolSchema.file,
            ContextSymbolSchema.category,
            ContextSymbolSchema.qname,
            ContextSymbolSchema.anno,
            ContextSymbolSchema.loop,
            ContextSymbolSchema.reassigned,
            ContextSymbolSchema.nested,
            ContextSymbolSchema.user_defined,
            ContextSymbolSchema.branching,
            ContextSymbolSchema.ctxt_category,
        ],
    )

    METADATA_DEPENDENCIES = (metadata.ScopeProvider,)

    def __init__(self, filepath: str, features: RelevantFeatures) -> None:
        super().__init__()

        self.features: RelevantFeatures = features

        self.scope_stack: list[tuple[str, ...]] = []
        self.branch_stack: list[cst.CSTNode] = []

        self.filepath = filepath
        self.loop_stack: list[cst.CSTNode] = []

        self.dfrs: list[ContextVectorVisitor.ContextVector] = []

        self._annassign_hinting: dict[str, cst.Annotation] = dict()


        self._noncst_metadata: dict[str, ] = dict()

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

    def visit_If(self, node: cst.If) -> None:
        self._enter_branch(node)

    def leave_If(self, _: cst.If) -> None:
        self._leave_branch()

    def visit_Else(self, node: cst.Else) -> None:
        self._enter_branch(node)

    def leave_Else(self, _: cst.Else) -> None:
        self._leave_branch()

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool | None:
        ident = _stringify(node.target)
        fullqual = ".".join((*self.scope(), ident))

        if node.value is not None:
            match self.get_metadata(metadata.ScopeProvider, node):
                case metadata.ClassScope():
                    category = TypeCollectionCategory.CLASS_ATTR

                case _:
                    category = TypeCollectionCategory.VARIABLE

            if fullqual in self._annassign_hinting:
                self._annassign_hinting.pop(fullqual)

            self._handle_annotatable(
                annotatable=node.target,
                identifier=ident,
                annotation=node.annotation,
                category=category,
            )

        else:
            self._annassign_hinting[fullqual] = node.annotation

    @m.call_if_inside(m.AssignTarget(target=m.Name() | m.Attribute(value=m.Name("self"))))
    def visit_AssignTarget(self, node: cst.AssignTarget) -> bool | None:
        self._visit_unannotated_target(node.target)

    @m.call_if_inside(m.AssignTarget())
    def visit_Tuple(self, node: cst.Tuple) -> bool | None:
        self._visit_unpackable(node.elements)

    @m.call_if_inside(m.AssignTarget())
    def visit_List(self, node: cst.List) -> bool | None:
        self._visit_unpackable(node.elements)

    def _visit_unpackable(self, elements: list[cst.BaseElement]) -> bool | None:
        targets = map(lambda e: e.value, elements)
        for target in filter(lambda e: not isinstance(e, (cst.Tuple, cst.List)), targets):
            self._visit_unannotated_target(target)

    def _visit_unannotated_target(self, target: cst.CSTNode) -> bool | None:
        name = get_full_name_for_node_or_raise(target)
        fullqual = self.qname_within_scope(name)

        # Consume stored hint if present
        hint = self._annassign_hinting.pop(fullqual, None)

        match self.get_metadata(metadata.ScopeProvider, target):
            case metadata.ClassScope():
                category = TypeCollectionCategory.CLASS_ATTR

            case _:
                category = TypeCollectionCategory.VARIABLE

        self._handle_annotatable(
            annotatable=target, identifier=_stringify(target), annotation=hint, category=category
        )

    def _update_scope_stack(self, name: str) -> None:
        self.scope_stack.append((*self.scope(), name))

    def _leave_scope(self) -> None:
        self.scope_stack.pop()

    def _enter_branch(self, node: cst.If | cst.Else) -> None:
        self.branch_stack.append(node)

    def _leave_branch(self) -> None:
        self.branch_stack.pop()

    def scope(self) -> tuple[str, ...]:
        return self.scope_stack[-1] if self.scope_stack else tuple()

    def qname_within_scope(self, identifier: str) -> str:
        return ".".join((*self.scope(), identifier))

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
        branching = int(self.features.branching and self._is_in_branch())

        categoryf = self._ctxt_category(annotatable, category)
        qname = self.qname_within_scope(identifier)

        self.dfrs.append(
            ContextVectorVisitor.ContextVector(
                self.filepath,
                category,
                qname,
                _stringify(annotation) or missing.NA,
                loopf,
                reassignedf,
                nestedf,
                user_definedf,
                branching,
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

    def _is_in_branch(self) -> bool:
        return bool(self.branch_stack)

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

    def build(self) -> pt.DataFrame[ContextSymbolSchema]:
        if not self.dfrs:
            return ContextSymbolSchema.example(size=0)

        wout_qname_ssa = [c for c in ContextSymbolSchemaColumns if c != "qname_ssa"]
        df = (
            pd.DataFrame(self.dfrs, columns=wout_qname_ssa)
            .pipe(generate_qname_ssas_for_file)
            .pipe(pt.DataFrame[ContextSymbolSchema])
        )

        # Reassigned variables also mean redefined parameters
        reass_variables = df[
            (df[ContextSymbolSchema.category] == TypeCollectionCategory.VARIABLE)
            & (df[ContextSymbolSchema.reassigned] == 1)
        ]
        # Find parameters by the same name
        parameters = df[ContextSymbolSchema.qname].isin(reass_variables[ContextSymbolSchema.qname])
        df.loc[parameters, ContextSymbolSchema.reassigned] = 1

        # Annotatables in flow control do not have to be marked as "reassigned" as long as there are no
        # occurrences of the same annotatable beforehand
        questionable_branching = (df[ContextSymbolSchema.branching] == 1) & (df[ContextSymbolSchema.reassigned] == 1)
        questionable_qnames = df.loc[questionable_branching, ContextSymbolSchema.qname]

        #false_reassigned = ...

        print(df)
        return df
