import builtins
import collections
import pathlib
import pydoc
import pandera.typing as pt
from common.schemas import ContextCategory, ContextSymbolSchema, TypeCollectionCategory
from common._helper import _stringify


import libcst as cst
import libcst.codemod as codemod
import libcst.metadata as metadata

from context.features import RelevantFeatures


class ContextVectorMaker(codemod.ContextAwareTransformer):
    METADATA_DEPENDENCIES = (metadata.ScopeProvider,)

    def __init__(self, context: codemod.CodemodContext, features: RelevantFeatures) -> None:
        super().__init__(context)

        self.features: RelevantFeatures = features

        self.scope_stack: list[tuple[str, ...]] = []
        self.scope_vars: collections.defaultdict[
            tuple[str, ...], set[str]
        ] = collections.defaultdict(set)

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

    def leave_FunctionDef(
        self, _: "cst.FunctionDef", updated_node: "cst.FunctionDef"
    ) -> cst.FunctionDef:
        self._leave_scope()
        return updated_node

    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
        self._update_scope_stack(node.name.value)

    def leave_ClassDef(self, _: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        self._leave_scope()
        return updated_node

    def visit_AssignTarget(self, node: cst.AssignTarget) -> bool | None:
        ident = _stringify(node.target)
        self._handle_annotatable(
            annotatable=node.target,
            identifier=_stringify(node.target),
            annotation=None,
            category=TypeCollectionCategory.VARIABLE,
        )
        self._update_scope_vars(ident)

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
        self._update_scope_vars(ident)

    def _update_scope_stack(self, name: str) -> None:
        self.scope_stack.append((*self.scope(), name))

    def _update_scope_vars(self, v: str) -> None:
        # TODO: Use recorded assignments instead; much easier to resolve
        scope = self.scope()
        slices = map(lambda o: scope[:o], range(len(scope)))

        if any(v in self.scope_vars[s] for s in slices):
            return
        self.scope_vars[scope].add(v)

    def _leave_scope(self) -> None:
        s = self.scope_stack.pop()
        del self.scope_vars[s]

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
        reassignedf = int(self.features.reassigned and self._is_reassigned(identifier))
        nestedf = int(self.features.nested and self._is_nested_scope(annotatable))
        user_definedf = int(self.features.user_defined and self._is_userdefined(annotation))

        categoryf = self._ctxt_category(annotatable, category)

        assert self.context.filename is not None
        assert self.context.metadata_manager is not None
        self.dfrs.append(
            (
                str(
                    pathlib.Path(self.context.filename).relative_to(
                        self.context.metadata_manager.root_path
                    )
                ),
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

    def _is_reassigned(self, identifier: str) -> bool:
        scope = self.scope()
        slices = map(lambda o: scope[:o], range(len(scope)))

        return any(identifier in self.scope_vars[s] for s in slices)

    def _is_nested_scope(self, n: cst.CSTNode) -> bool:
        # Detect class in class or function in function
        scopes = [scope := self.get_metadata(metadata.ScopeProvider, n)]
        while not (scope is None or isinstance(scope, metadata.BuiltinScope)):
            scopes.append(scope := scope.parent)

        return len(set(map(type, scopes))) < len(scopes)

    def _is_userdefined(self, annotation: cst.Annotation | None) -> bool:
        if annotation is None:
            return False

        if (loc := pydoc.locate(_stringify(annotation.annotation))) is None:
            return False

        return loc.__qualname__ in dir(builtins)

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
