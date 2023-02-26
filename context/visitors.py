import builtins
import collections
import pathlib

import libcst
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
from common import visitors
from context.features import RelevantFeatures


def generate_context_vectors_for_file(
    features: RelevantFeatures, repo: pathlib.Path, path: pathlib.Path
) -> pt.DataFrame[ContextSymbolSchema]:
    visitor = ContextVectorVisitor(filepath=str(path.relative_to(repo)), features=features)
    module = libcst.parse_module(path.open().read())

    md = metadata.MetadataWrapper(module)
    md.visit(visitor)

    return visitor.build()


class ContextVectorVisitor(
    visitors.HintableReturnVisitor,
    visitors.HintableParameterVisitor,
    visitors.HintableDeclarationVisitor,
):
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

    METADATA_DEPENDENCIES = (metadata.ScopeProvider, metadata.ParentNodeProvider)

    def __init__(self, filepath: str, features: RelevantFeatures) -> None:
        super().__init__()

        self.features: RelevantFeatures = features

        # parentage AST-tree; includes every node that can have children with bodies,
        # e.g. ClassDef, FunctionDef, For, If, etc.
        self.full_scope_nodes: list[libcst.CSTNode] = []

        # names of nodes that builds parentage; 1 to 1 index-based correspondence
        # to self.full_scope_nodes
        self.full_scope_names: list[tuple[str, ...]] = []

        # scope AST-tree; includes every node that influences scope.
        # Limited to ClassDef, FunctionDef
        self.real_scope_names: list[libcst.CSTNode] = []

        # mapping of self.real_scope_names to symbols declared therein
        # used for reassignment checks
        self.visible_symbols: collections.defaultdict[
            tuple[str, ...], set[str]
        ] = collections.defaultdict(set)

        # mapping of self.real_scope_names to symbols declared therein;
        # this differs from self.visible_symbols in that it is NOT used for reassignment checking
        # this is useful for when symbols aren't visible to children of parents,
        # e.g. libcst.If -> libcst.Else
        self.invisible_symbols: collections.defaultdict[
            tuple[str, ...], set[str]
        ] = collections.defaultdict(set)

        self.filepath = filepath
        self._annassign_hinting: dict[str, libcst.Annotation] = dict()

        self.dfrs: list[ContextVectorVisitor.ContextVector] = []

    def annotated_function(
        self, function: libcst.FunctionDef, annotation: libcst.Annotation
    ) -> None:
        self._handle_annotatable(
            annotatable=function,
            identifier=function.name.value,
            annotation=annotation,
            category=TypeCollectionCategory.CALLABLE_RETURN,
        )

    def unannotated_function(self, function: libcst.FunctionDef) -> None:
        self._handle_annotatable(
            annotatable=function,
            identifier=function.name.value,
            annotation=None,
            category=TypeCollectionCategory.CALLABLE_RETURN,
        )

    def annotated_param(self, param: libcst.Param, annotation: libcst.Annotation) -> None:
        self._handle_annotatable(
            annotatable=param,
            identifier=param.name.value,
            annotation=annotation,
            category=TypeCollectionCategory.CALLABLE_PARAMETER,
        )

    def unannotated_param(self, param: libcst.Param) -> None:
        self._handle_annotatable(
            annotatable=param,
            identifier=param.name.value,
            annotation=None,
            category=TypeCollectionCategory.CALLABLE_PARAMETER,
        )

    def instance_attribute_hint(
        self, target: libcst.Name, annotation: libcst.Annotation | None
    ) -> None:
        self._handle_annotatable(
            annotatable=target,
            identifier=target.value,
            annotation=annotation,
            category=TypeCollectionCategory.INSTANCE_ATTR,
        )

    def annotated_assignment(
        self, target: libcst.Name | libcst.Attribute, annotation: libcst.Annotation
    ) -> None:
        ident = get_full_name_for_node_or_raise(target)
        fullqual = self.qname_within_scope(ident)

        self._annassign_hinting.pop(fullqual, None)

        self._handle_annotatable(
            annotatable=target,
            identifier=ident,
            annotation=annotation,
            category=TypeCollectionCategory.VARIABLE,
        )

    def annotated_hint(
        self, target: libcst.Name | libcst.Attribute, annotation: libcst.Annotation
    ) -> None:
        ident = get_full_name_for_node_or_raise(target)
        fullqual = self.qname_within_scope(ident)

        self._annassign_hinting[fullqual] = annotation

    def unannotated_target(self, target: libcst.Name | libcst.Attribute) -> None:
        name = get_full_name_for_node_or_raise(target)
        fullqual = self.qname_within_scope(name)

        # Reference stored hint if present
        hint = self._annassign_hinting.get(fullqual, None)
        self._handle_annotatable(
            annotatable=target,
            identifier=name,
            annotation=hint,
            category=TypeCollectionCategory.VARIABLE,
        )

    @m.visit(m.If() | m.Else())
    def _enter_branch(self, branch: libcst.If | libcst.Else):
        self.full_scope_nodes.append(branch)
        self.full_scope_names.append(tuple((*self.scope_components(), branch.__class__.__name__)))
        self.visible_symbols[self.scope_components()] = set()

    @m.leave(m.If() | m.Else())
    def _leave_branch(self, branch: libcst.If | libcst.Else):
        *outer, _ = leaving = self.scope_components()

        # Branches attached to non-branching nodes
        if (
            len(self.full_scope_nodes) >= 2  # access safety, should be guaranteed though
            and m.matches(
                self.full_scope_nodes[-2], (m.While | m.For | m.Try | m.TryStar)(orelse=m.Else())
            )  # is and if with a child branch
            and self.full_scope_nodes[-2].orelse
            is branch  # this node's orelse child is precisely this branch
        ):
            # Assume else-execution is guaranteed at some point -> set-union;
            # otherwise not much point for it to exist
            self.visible_symbols[tuple(outer)] |= self.invisible_symbols.pop(leaving, set())

        # "Real" branching begins here:

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

        # Initiating branch; move invisible symbols to visible symbols
        # Single If; set-intersection; symbol may be unbound
        elif m.matches(branch, m.If(orelse=~(m.If() | m.Else()))):
            self.visible_symbols[tuple(outer)] &= self.invisible_symbols.pop(leaving, set())

        # If with further branches; set-union of intersected symbols
        elif m.matches(branch, m.If(orelse=m.If() | m.Else())):
            self.visible_symbols[tuple(outer)] |= self.invisible_symbols.pop(leaving, set())

        else:
            assert False, libcst.Module([branch]).code

        self.full_scope_nodes.pop()
        self.full_scope_names.pop()

    @m.visit(m.Try() | m.TryStar() | m.ExceptHandler() | m.Finally())
    def _enter_exception_block(self, block: libcst.Try | libcst.TryStar | libcst.ExceptHandler | libcst.Finally):
        self.full_scope_nodes.append(block)
        self.full_scope_names.append(tuple((*self.scope_components(), block.__class__.__name__)))
        self.visible_symbols[self.scope_components()] = set()

    @m.leave(m.Try() | m.TryStar() | m.ExceptHandler() | m.Finally())
    def _leave_exception_block(self, _: libcst.Try | libcst.TryStar | libcst.ExceptHandler | libcst.Finally):
        *outer, _ = leaving = self.scope_components()

        # Each body's entrance and exit points can be triggered at any point; 
        # simply assume latest possible execution, i.e.
        # propogate all symbols declared in bodies
        self.visible_symbols[tuple(outer)] |= self.visible_symbols.pop(leaving, set())

        self.full_scope_nodes.pop()
        self.full_scope_names.pop()

    def _handle_annotatable(
        self,
        annotatable: libcst.CSTNode,
        identifier: str,
        annotation: libcst.Annotation | None,
        category: TypeCollectionCategory,
    ) -> None:
        reassignedf = int(self.features.reassigned and self._is_reassigned(identifier))

        self.visible_symbols[self.scope_components()].add(identifier)

        loopf = int(self.features.loop and self._is_in_loop(annotatable))
        nestedf = int(self.features.nested and self._is_nested_scope(annotatable))
        user_definedf = int(self.features.user_defined and self._is_userdefined(annotation))
        branching = int(self.features.branching and self._is_in_branch())

        categoryf = self._ctxt_category(category)
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

    def _is_in_loop(self, annotatable: libcst.CSTNode) -> bool:
        # Break out of unpackable
        # while isinstance((parent := self.get_metadata(metadata.ParentNodeProvider, annotatable)), libcst.Tuple | libcst.List):
        #    ...


        return any(isinstance(s, libcst.For | libcst.While) for s in self.full_scope_nodes)

    def _is_reassigned(self, identifier: str) -> bool:
        scope = self.scope_components()

        for window in reversed(range(len(scope))):
            window_scope = scope[: window + 1]
            if identifier in self.visible_symbols.get(window_scope, set()):
                return True

            if isinstance(self.full_scope_nodes[window], libcst.FunctionDef | libcst.ClassDef):
                return False

        return identifier in self.visible_symbols.get((), set())

    def _is_nested_scope(self, node: libcst.CSTNode) -> bool:
        # Detect class in class or function in function
        scopes = [scope := self.get_metadata(metadata.ScopeProvider, node)]
        while not (scope is None or isinstance(scope, metadata.BuiltinScope)):
            scopes.append(scope := scope.parent)

        counted = collections.Counter(map(type, scopes))

        fncount = counted.get(metadata.FunctionScope, 0) + isinstance(node, libcst.FunctionDef)
        czcount = counted.get(metadata.ClassScope, 0) + isinstance(node, metadata.ClassScope)
        return fncount >= 2 or czcount >= 2

    def _is_in_branch(self) -> bool:
        return any(isinstance(s, libcst.If | libcst.Else) for s in self.full_scope_nodes)

    def _is_userdefined(self, annotation: libcst.Annotation | None) -> bool:
        if annotation is None:
            return False

        a = _stringify(annotation.annotation)
        sanitised = "".join(a.split())
        unions = sanitised.split("|")

        return any(u not in dir(builtins) for u in unions)

    @m.visit(m.FunctionDef() | m.ClassDef())
    def _enter_scope(self, node: libcst.FunctionDef | libcst.ClassDef) -> None:
        self.full_scope_nodes.append(node)

        self.full_scope_names.append(tuple((*self.scope_components(), node.name.value)))
        self.real_scope_names.append(tuple((*self.real_scope_components(), node.name.value)))

        self.visible_symbols[self.scope_components()] = set()

    @m.leave(m.FunctionDef() | m.ClassDef())
    def _leave_scope(self, _: libcst.FunctionDef | libcst.ClassDef) -> None:
        del self.visible_symbols[self.scope_components()]
        self.full_scope_nodes.pop()

        self.full_scope_names.pop()
        self.real_scope_names.pop()

    @m.visit(m.While() | m.For() | m.CompFor())
    def _enter_loop(self, node: libcst.While | libcst.For | libcst.CompFor) -> None:
        self.full_scope_nodes.append(node)
        self.full_scope_names.append(tuple((*self.scope_components(), node.__class__.__name__)))

        self.visible_symbols[self.scope_components()] = set()


    @m.leave(m.While() | m.For() | m.CompFor())
    def _leave_loop(self, _: libcst.While | libcst.For | libcst.CompFor) -> None:
        # Symbols declared here persist into lower scope, merge them in
        *outer, _ = leaving = self.scope_components()
        self.visible_symbols[tuple(outer)] |= self.visible_symbols.pop(leaving, set())

        self.full_scope_nodes.pop()
        self.full_scope_names.pop() 


    def leave_If_body(self, _: libcst.If) -> None:
        self._leave_branch_body()

    def leave_Else_body(self, _: libcst.Else) -> None:
        self._leave_branch_body()

    def _leave_branch_body(self) -> None:
        # Move newly tracked symbols to invisible symbols
        leaving = self.scope_components()
        self.invisible_symbols[leaving] = self.visible_symbols.pop(leaving, set())


    def scope_components(self) -> tuple[str, ...]:
        return self.full_scope_names[-1] if self.full_scope_names else tuple()

    def real_scope_components(self) -> tuple[str, ...]:
        return self.real_scope_names[-1] if self.real_scope_names else tuple()

    def qname_within_scope(self, identifier: str) -> str:
        return ".".join((*self.real_scope_components(), identifier))

    def _ctxt_category(self, category: TypeCollectionCategory) -> ContextCategory:
        match category:
            case TypeCollectionCategory.CALLABLE_RETURN:
                return ContextCategory.CALLABLE_RETURN

            case TypeCollectionCategory.CALLABLE_PARAMETER:
                return ContextCategory.CALLABLE_PARAMETER

            case TypeCollectionCategory.INSTANCE_ATTR:
                return ContextCategory.INSTANCE_ATTR

            case TypeCollectionCategory.VARIABLE:
                return ContextCategory.VARIABLE

    def build(self) -> pt.DataFrame[ContextSymbolSchema]:
        if not self.dfrs:
            return ContextSymbolSchema.example(size=0)

        df = (
            pd.DataFrame(self.dfrs, columns=ContextVectorVisitor.ContextVector._fields)
            .pipe(generate_qname_ssas_for_file)
            .pipe(pt.DataFrame[ContextSymbolSchema])
        )
        return df
