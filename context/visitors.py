import builtins
import collections
import pathlib

import libcst as cst
import libcst.metadata as metadata
import pandas as pd
import pandera.typing as pt
from libcst import matchers as m
from libcst.helpers import get_full_name_for_node_or_raise
from pandas._libs import missing

from common.ast_helper import _stringify, generate_qname_ssas_for_file
from common.schemas import (
    ContextCategory,
    ContextSymbolSchema,
    TypeCollectionCategory,
)
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

        # parentage AST-tree; includes every node that can have children with bodies,
        # e.g. ClassDef, FunctionDef, For, If, etc.
        self.full_scope_nodes: list[cst.CSTNode] = []

        # names of nodes that builds parentage; 1 to 1 index-based correspondence
        # to self.full_scope_nodes
        self.full_scope_names: list[tuple[str, ...]] = []

        # scope AST-tree; includes every node that influences scope.
        # Limited to ClassDef, FunctionDef
        self.real_scope_names: list[cst.CSTNode] = []

        # mapping of self.real_scope_names to symbols declared therein
        # used for reassignment checks
        self.visible_symbols: collections.defaultdict[
            tuple[str, ...], set[str]
        ] = collections.defaultdict(set)

        # mapping of self.real_scope_names to symbols declared therein;
        # this differs from self.visible_symbols in that it is NOT used for reassignment checking
        # this is useful for when symbols aren't visible to children of parents,
        # e.g. cst.If -> cst.Else
        self.invisible_symbols: collections.defaultdict[
            tuple[str, ...], set[str]
        ] = collections.defaultdict(set)

        self.filepath = filepath
        self._annassign_hinting: dict[str, cst.Annotation] = dict()

        self.dfrs: list[ContextVectorVisitor.ContextVector] = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        self._handle_annotatable(
            annotatable=node,
            identifier=node.name.value,
            annotation=node.returns,
            category=TypeCollectionCategory.CALLABLE_RETURN,
        )

        self._enter_scope(node, node.name.value)

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
        self._enter_scope(node, node.name.value)

    def leave_ClassDef(self, _: cst.ClassDef) -> None:
        self._leave_scope()

    def visit_If(self, node: cst.If) -> None:
        self._enter_branch(node)

    def leave_If_body(self, _: cst.If) -> None:
        self._leave_branch_body()

    def leave_If(self, node: cst.If) -> None:
        self._leave_branch(node)

    def visit_Else(self, node: cst.Else) -> None:
        self._enter_branch(node)

    def leave_Else_body(self, _: cst.Else) -> None:
        self._leave_branch_body()

    def leave_Else(self, node: cst.Else) -> None:
        self._leave_branch(node)

    def visit_While(self, node: cst.While) -> bool | None:
        self._enter_loop(node)

    def leave_While(self, _: cst.While) -> None:
        self._leave_loop()

    @m.call_if_inside(
        m.AnnAssign(target=m.Name() | m.Attribute(value=m.Name("self"), attr=m.Name()))
    )
    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool | None:
        ident = _stringify(node.target)
        fullqual = self.qname_within_scope(ident)

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
    
    
    @m.call_if_inside(
        m.AssignTarget(
            target=m.Name()
            | m.Attribute(value=m.Name("self"), attr=m.Name())
            | m.List()
            | m.Tuple()
        )
    )
    def visit_AssignTarget(self, node: cst.AssignTarget) -> bool | None:
        if m.matches(node.target, m.Name() | m.Attribute(value=m.Name("self"), attr=m.Name())):
            self._visit_unannotated_target(node.target)
        elif m.matches(node.target, m.List() | m.Tuple()):
            self._visit_unpackable(node.target)


    @m.call_if_inside(
        m.AugAssign(
            target=m.Name()
            | m.Attribute(value=m.Name("self"), attr=m.Name())
            | m.List()
            | m.Tuple()
        )
    )
    def visit_AugAssign(self, node: cst.AugAssign) -> bool | None:
        if m.matches(node.target, m.Name() | m.Attribute(value=m.Name("self"), attr=m.Name())):
            self._visit_unannotated_target(node.target)
        elif m.matches(node.target, m.List() | m.Tuple()):
            self._visit_unpackable(node.target)

    def visit_WithItem(self, node: cst.WithItem) -> bool | None:
        if node.asname is not None:
            if m.matches(node.asname.name, m.Name()):
                self._visit_unannotated_target(node.asname.name)
            elif m.matches(node.asname.name, m.List() | m.Tuple()):
                self._visit_unpackable(node.asname.name)

    def visit_For(self, node: cst.For) -> bool | None:
        self._enter_loop(node)

        if m.matches(node.target, m.Name() | m.Attribute(value=m.Name("self"), attr=m.Name())):
            self._visit_unannotated_target(node.target)
        elif m.matches(node.target, m.List() | m.Tuple()):
            self._visit_unpackable(node.target)

    def leave_For(self, _: cst.For) -> None:
        self._leave_loop()

    def visit_CompFor(self, node: cst.CompFor) -> bool | None:
        if m.matches(node.target, m.Name() | m.Attribute(value=m.Name("self"), attr=m.Name())):
            self._visit_unannotated_target(node.target)
        elif m.matches(node.target, m.List() | m.Tuple()):
            self._visit_unpackable(node.target)

    def visit_NamedExpr(self, node: cst.NamedExpr) -> bool | None:
        if m.matches(node.target, m.Name() | m.Attribute(value=m.Name("self"), attr=m.Name())):
            self._visit_unannotated_target(node.target)
        elif m.matches(node.target, m.List() | m.Tuple()):
            self._visit_unpackable(node.target)

    def _visit_unpackable(self, unpackable: cst.List | cst.Tuple) -> bool | None:
        targets = map(lambda e: e.value, unpackable.elements)
        for target in targets:
            if m.matches(target, m.Tuple() | m.List()):
                self._visit_unpackable(target)
            else:
                self._visit_unannotated_target(target)

    def _visit_unannotated_target(self, target: cst.CSTNode) -> bool | None:
        name = get_full_name_for_node_or_raise(target)
        fullqual = self.qname_within_scope(name)

        # Reference stored hint if present
        hint = self._annassign_hinting.get(fullqual, None)

        match self.get_metadata(metadata.ScopeProvider, target):
            case metadata.ClassScope():
                category = TypeCollectionCategory.CLASS_ATTR

            case _:
                category = TypeCollectionCategory.VARIABLE

        self._handle_annotatable(
            annotatable=target, identifier=_stringify(target), annotation=hint, category=category
        )

    def _handle_annotatable(
        self,
        annotatable: cst.CSTNode,
        identifier: str,
        annotation: cst.Annotation | None,
        category: TypeCollectionCategory,
    ) -> None:
        reassignedf = int(self.features.reassigned and self._is_reassigned(identifier))

        if not isinstance(annotatable, cst.FunctionDef | cst.ClassDef):
            self.visible_symbols[self.scope_components()].add(identifier)

        loopf = int(self.features.loop and self._is_in_loop())
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
        return any(isinstance(s, cst.For | cst.While) for s in self.full_scope_nodes)

    def _is_reassigned(self, identifier: str) -> bool:
        scope = self.scope_components()

        for window in reversed(range(len(scope))):
            window_scope = scope[: window + 1]
            if identifier in self.visible_symbols.get(window_scope, set()):
                return True

            if isinstance(self.full_scope_nodes[window], cst.FunctionDef | cst.ClassDef):
                return False

        return False

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
        return any(isinstance(s, cst.If | cst.Else) for s in self.full_scope_nodes)

    def _is_userdefined(self, annotation: cst.Annotation | None) -> bool:
        if annotation is None:
            return False

        a = _stringify(annotation.annotation)
        sanitised = "".join(a.split())
        unions = sanitised.split("|")

        return any(u not in dir(builtins) for u in unions)

    def _enter_scope(self, node: cst.CSTNode, name: str) -> None:
        self.full_scope_nodes.append(node)

        self.full_scope_names.append(tuple((*self.scope_components(), name)))
        if m.matches(node, m.FunctionDef() | m.ClassDef()):
            self.real_scope_names.append(tuple((*self.real_scope_components(), name)))

        self.visible_symbols[self.scope_components()] = set()

    def _enter_branch(self, node: cst.If | cst.Else) -> None:
        self.full_scope_nodes.append(node)
        self.full_scope_names.append(tuple((*self.scope_components(), node.__class__.__name__)))
        self.visible_symbols[self.scope_components()] = set()

    def _enter_loop(self, node: cst.While | cst.For) -> None:
        self.full_scope_nodes.append(node)
        self.full_scope_names.append(tuple((*self.scope_components(), node.__class__.__name__)))
        self.visible_symbols[self.scope_components()] = set()

    def _leave_scope(self) -> None:
        del self.visible_symbols[self.scope_components()]

        self.real_scope_names.pop()
        self.full_scope_nodes.pop()
        self.full_scope_names.pop()

    def _leave_branch_body(self) -> None:
        # Move newly tracked symbols to invisible symbols
        leaving = self.scope_components()
        self.invisible_symbols[leaving] = self.visible_symbols.pop(leaving, set())

    def _leave_branch(self, branch: cst.If | cst.Else) -> None:
        *outer, _ = leaving = self.scope_components()

        # Special-case: For-Else; set-intersection; symbol may be unbound
        if (
            len(self.full_scope_nodes) >= 2  # access safety
            and m.matches(
                self.full_scope_nodes[-2], m.For(orelse=m.Else())
            )  # is and if with a child branch
            and self.full_scope_nodes[-2].orelse
            is branch  # this if's child is precisely this branch
        ):
            self.visible_symbols[tuple(outer)] &= self.invisible_symbols.pop(leaving, set())

        # propagate set-intersection of invisible symbols upwards to parent branch, i.e.
        # if True
        #   ...
        # else: <--
        #   ...
        elif (
            len(self.full_scope_nodes) >= 2  # access safety
            and m.matches(
                self.full_scope_nodes[-2], m.If(orelse=m.If() | m.Else())
            )  # is and if with a child branch
            and self.full_scope_nodes[-2].orelse
            is branch  # this if's child is precisely this branch
        ):
            self.invisible_symbols[tuple(outer)] &= self.invisible_symbols.pop(leaving, set())

        # otherwise we reached first branch; move invisible symbols to visible symbols
        # Single If; set-intersection; symbol may be unbound
        elif m.matches(self.full_scope_nodes[-1], m.If(orelse=~(m.If() | m.Else()))):
            self.visible_symbols[tuple(outer)] &= self.invisible_symbols.pop(leaving, set())

        # If with further branches; set-union of intersected symbols
        elif m.matches(self.full_scope_nodes[-1], m.If(orelse=m.If() | m.Else())):
            self.visible_symbols[tuple(outer)] |= self.invisible_symbols.pop(leaving, set())

        else:
            assert False, cst.Module([branch]).code

        self.full_scope_nodes.pop()
        self.full_scope_names.pop()

    def _leave_loop(self) -> None:
        # Symbols declared here persist into lower scope, merge them in
        *outer, _ = leaving = self.scope_components()
        self.visible_symbols[tuple(outer)] |= self.visible_symbols.pop(leaving, set())

        self.full_scope_nodes.pop()
        self.full_scope_names.pop()

    def scope_components(self) -> tuple[str, ...]:
        return self.full_scope_names[-1] if self.full_scope_names else tuple()

    def real_scope_components(self) -> tuple[str, ...]:
        return self.real_scope_names[-1] if self.real_scope_names else tuple()

    def qname_within_scope(self, identifier: str) -> str:
        return ".".join((*self.real_scope_components(), identifier))

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

        df = (
            pd.DataFrame(self.dfrs, columns=ContextVectorVisitor.ContextVector._fields)
            .pipe(generate_qname_ssas_for_file)
            .pipe(pt.DataFrame[ContextSymbolSchema])
        )
        return df
