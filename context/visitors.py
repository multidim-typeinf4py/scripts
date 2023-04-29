import builtins
import collections
import pathlib
import typing
from typing import Union, Optional

import libcst
import pandas as pd
import pandera.typing as pt
import tqdm
from libcst import metadata, codemod as c, matchers as m
from libcst.helpers import get_full_name_for_node_or_raise
from pandas._libs import missing
from tqdm.contrib.concurrent import process_map

from common import visitors
from common._traversal import T
from common.ast_helper import _stringify, generate_qname_ssas_for_file
from common.metadata import anno4inst
from common.schemas import (
    ContextCategory,
    ContextSymbolSchema,
    TypeCollectionCategory,
)
from context.features import RelevantFeatures
from utils import worker_count


def generate_context_vectors_for_project(
    features: RelevantFeatures, repo: pathlib.Path
) -> pt.DataFrame[ContextSymbolSchema]:
    repo_root = str(repo)
    assert repo.is_dir(), f"Path to folder is required, got {repo_root}"
    files = c.gather_files([repo_root], include_stubs=False)

    file2code = {
        file: open(file).read()
        for file in tqdm.tqdm(files, desc=f"Loading files for dataset creating of {repo_root}")
    }

    collector = generate_context_vectors_for_file(
        features=features, repo_root=repo_root, files=files
    )
    collections = process_map(
        collector,
        file2code.items(),
        total=len(file2code),
        desc=f"Creating dataset for {repo_root}",
        position=0,
        max_workers=worker_count()
    )
    return pd.concat(collections, ignore_index=True).pipe(pt.DataFrame[ContextSymbolSchema])


def generate_context_vectors_for_file(
    features: RelevantFeatures, repo: pathlib.Path, file2code: tuple[pathlib.Path, str]
) -> pt.DataFrame[ContextSymbolSchema]:
    path, code = file2code
    module = libcst.parse_module(code)

    md = metadata.MetadataWrapper(module)

    visitor = ContextVectorVisitor(filepath=str(path.relative_to(repo)), features=features)
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
            ContextSymbolSchema.simple_name,
            ContextSymbolSchema.anno,
            ContextSymbolSchema.loop,
            ContextSymbolSchema.reassigned,
            ContextSymbolSchema.nested,
            ContextSymbolSchema.builtin,
            ContextSymbolSchema.branching,
            ContextSymbolSchema.ctxt_category,
        ],
    )

    METADATA_DEPENDENCIES = (
        metadata.ScopeProvider,
        metadata.ParentNodeProvider,
        anno4inst.Annotation4InstanceProvider,
    )

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
        self.real_scope_names: list[tuple[str, ...]] = []

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

        self.keyword_modified_targets: set[str] = set()

        self.filepath = filepath

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

    def instance_attribute_hint(self, _: libcst.AnnAssign, target: libcst.Name) -> None:
        self._handle_annotatable(
            annotatable=target,
            identifier=target.value,
            annotation=None,
            category=TypeCollectionCategory.VARIABLE,
        )

    def libsa4py_hint(self, original_node: libcst.Assign, target: libcst.Name) -> None:
        self._handle_annotatable(
            annotatable=target,
            identifier=target.value,
            annotation=None,
            category=TypeCollectionCategory.VARIABLE,
        )

    def annotated_hint(
        self, original_node: libcst.AnnAssign, target: Union[libcst.Name, libcst.Attribute]
    ) -> T:
        pass

    def annotated_assignment(
        self, original_node: libcst.AnnAssign, target: Union[libcst.Name, libcst.Attribute]
    ) -> T:
        self.handle_variable_target(target)

    def unannotated_assign_single_target(
        self,
        original_node: libcst.Assign,
        target: Union[libcst.Name, libcst.Attribute],
    ) -> None:
        return self.handle_variable_target(
            target,
        )

    def unannotated_assign_multiple_targets(
        self,
        original_node: Union[libcst.Assign, libcst.AugAssign],
        target: Union[libcst.Name, libcst.Attribute],
    ) -> None:
        return self.handle_variable_target(target)

    def for_target(self, original_node: libcst.For, target: Union[libcst.Name, libcst.Attribute]) -> None:
        return self.handle_variable_target(target)

    def withitem_target(
        self, original_node: libcst.With, target: Union[libcst.Name, libcst.Attribute]
    ) -> None:
        return self.handle_variable_target(target)

    def handle_variable_target(self, target: Union[libcst.Name, libcst.Attribute]) -> None:
        name = get_full_name_for_node_or_raise(target)

        annotation = self.get_metadata(anno4inst.Annotation4InstanceProvider, target).labelled

        # Reference stored hint if present
        self._handle_annotatable(
            annotatable=target,
            identifier=name,
            annotation=annotation,
            category=TypeCollectionCategory.VARIABLE,
        )

    def global_target(
        self,
        _: Union[libcst.Assign, libcst.AnnAssign, libcst.AugAssign],
        target: libcst.Name,
    ) -> None:
        self.scope_overwritten_target(target)

    def nonlocal_target(
        self,
        _: Union[libcst.Assign, libcst.AnnAssign, libcst.AugAssign],
        target: libcst.Name,
    ) -> None:
        self.scope_overwritten_target(target)

    def scope_overwritten_target(self, target: libcst.Name) -> None:
        self.keyword_modified_targets.add(get_full_name_for_node_or_raise(target))

    @m.visit(m.If() | m.Else())
    def _enter_branch(self, branch: Union[libcst.If, libcst.Else]):
        self.full_scope_nodes.append(branch)
        self.full_scope_names.append(tuple((*self.scope_components(), branch.__class__.__name__)))
        self.visible_symbols[self.scope_components()] = set()

    @m.leave(m.If() | m.Else())
    def _leave_branch(self, branch: Union[libcst.If, libcst.Else]):
        *outer, _ = leaving = self.scope_components()

        # Branches attached to non-branching nodes
        if (
            len(self.full_scope_nodes) >= 2  # access safety, should be guaranteed though
            and m.matches(
                self.full_scope_nodes[-2],
                (m.While | m.For | m.Try | m.TryStar)(orelse=m.Else()),
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

    @m.visit(m.Try() | m.TryStar() | m.ExceptHandler() | m.ExceptStarHandler() | m.Finally())
    def _enter_exception_block(
        self,
        block: Union[libcst.Try, libcst.TryStar, libcst.ExceptHandler, libcst.ExceptStarHandler, libcst.Finally],
    ):
        self.full_scope_nodes.append(block)
        self.full_scope_names.append(tuple((*self.scope_components(), block.__class__.__name__)))
        self.visible_symbols[self.scope_components()] = set()

    @m.leave(m.Try() | m.TryStar() | m.ExceptHandler() | m.Finally())
    def _leave_exception_block(
        self, _: Union[libcst.Try, libcst.TryStar, libcst.ExceptHandler, libcst.Finally]
    ):
        *outer, _ = leaving = self.scope_components()

        # Each body's entrance and exit points can be triggered at any point;
        # simply assume latest possible execution, i.e.
        # propagate all symbols declared in bodies
        self.visible_symbols[tuple(outer)] |= self.visible_symbols.pop(leaving, set())

        self.full_scope_nodes.pop()
        self.full_scope_names.pop()

    def _handle_annotatable(
        self,
        annotatable: Union[libcst.Name, libcst.Attribute, libcst.FunctionDef, libcst.Param],
        identifier: str,
        annotation: Optional[libcst.Annotation],
        category: TypeCollectionCategory,
    ) -> None:
        reassignedf = int(self.features.reassigned and self._is_reassigned(identifier))

        if m.matches(annotatable, m.Name()):
            simple_name = annotatable.value
        elif m.matches(annotatable, m.Attribute()):
            simple_name = annotatable.attr.value
        elif m.matches(annotatable, m.FunctionDef()):
            simple_name = annotatable.name.value
        elif m.matches(annotatable, m.Param()):
            simple_name = annotatable.name.value

        self.visible_symbols[self.scope_components()].add(identifier)

        loopf = int(self.features.loop and self._is_in_loop(annotatable))
        nestedf = int(self.features.nested and self._is_nested_scope(annotatable))
        builtinf = int(self.features.builtin and self.is_builtin(annotation))
        branching = int(self.features.branching and self._is_in_branch())

        categoryf = self._ctxt_category(category)
        qname = self.qname_within_scope(identifier)


        self.dfrs.append(
            ContextVectorVisitor.ContextVector(
                self.filepath,
                category,
                qname,
                simple_name,
                _stringify(annotation) or missing.NA,
                loopf,
                reassignedf,
                nestedf,
                builtinf,
                branching,
                categoryf,
            )
        )

    def _is_in_loop(self, _: libcst.CSTNode) -> bool:
       return any(isinstance(s, (libcst.For, libcst.While)) for s in self.full_scope_nodes)

    def _is_reassigned(self, identifier: str) -> bool:
        scope = self.scope_components()

        for window in reversed(range(len(scope))):
            window_scope = scope[: window + 1]
            if identifier in self.visible_symbols.get(window_scope, set()):
                return True

            if isinstance(self.full_scope_nodes[window], (libcst.FunctionDef, libcst.ClassDef)):
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
        return any(isinstance(s, (libcst.If, libcst.Else)) for s in self.full_scope_nodes)

    def is_builtin(self, annotation: Optional[libcst.Annotation]) -> bool:
        if annotation is None:
            return False

        if isinstance(annotation.annotation, libcst.Subscript):
            annotation = annotation.annotation.value
        else:
            annotation = annotation.annotation

        if isinstance(annotation, libcst.Name):
            ty = annotation.value
        else:
            ty = annotation.attr.value

        return ty in dir(builtins) or ty in dir(typing)

    @m.visit(m.FunctionDef() | m.ClassDef())
    def _enter_scope(self, node: Union[libcst.FunctionDef, libcst.ClassDef]) -> None:
        self.full_scope_nodes.append(node)

        self.full_scope_names.append(tuple((*self.scope_components(), node.name.value)))
        self.real_scope_names.append(tuple((*self.real_scope_components(), node.name.value)))

        self.visible_symbols[self.scope_components()] = set()

    @m.leave(m.FunctionDef() | m.ClassDef())
    def _leave_scope(self, _: Union[libcst.FunctionDef, libcst.ClassDef]) -> None:
        del self.visible_symbols[self.scope_components()]
        self.full_scope_nodes.pop()

        self.full_scope_names.pop()
        self.real_scope_names.pop()

    @m.visit(m.While() | m.For() | m.CompFor())
    def _enter_loop(self, node: Union[libcst.While, libcst.For, libcst.CompFor]) -> None:
        self.full_scope_nodes.append(node)
        self.full_scope_names.append(tuple((*self.scope_components(), node.__class__.__name__)))

        self.visible_symbols[self.scope_components()] = set()

    @m.leave(m.While() | m.For() | m.CompFor())
    def _leave_loop(self, _: Union[libcst.While, libcst.For, libcst.CompFor]) -> None:
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
        if category is TypeCollectionCategory.CALLABLE_RETURN:
            return ContextCategory.CALLABLE_RETURN

        if category is TypeCollectionCategory.CALLABLE_PARAMETER:
            return ContextCategory.CALLABLE_PARAMETER

        if category is TypeCollectionCategory.VARIABLE:
            return ContextCategory.VARIABLE


    def build(self) -> pt.DataFrame[ContextSymbolSchema]:
        if not self.dfrs:
            return ContextSymbolSchema.example(size=0)

        df = (
            pd.DataFrame(self.dfrs, columns=ContextVectorVisitor.ContextVector._fields)
            .pipe(generate_qname_ssas_for_file)
            .pipe(pt.DataFrame[ContextSymbolSchema])
        )

        # Update keyword modified scopage
        reassigned_names = df[ContextSymbolSchema.qname].isin(self.keyword_modified_targets)
        df.loc[reassigned_names, ContextSymbolSchema.reassigned] = 1

        return df
