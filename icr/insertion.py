import functools
import pathlib

from libcst import codemod
import libcst as cst
from libcst.codemod.visitors._add_imports import AddImportsVisitor
from libcst.codemod.visitors._apply_type_annotations import ApplyTypeAnnotationsVisitor
from libcst import codemod
import libcst.matchers as m

import pandera.typing as pt

from common._helper import _stringify
from common.schemas import TypeCollectionSchema
from common.storage import TypeCollection


class _ParameterHintRemover(cst.CSTTransformer):
    def leave_Param(self, _: cst.Param, updated_node: cst.Param) -> cst.Param:
        return updated_node.with_changes(annotation=None)


class _ReturnHintRemover(cst.CSTTransformer):
    def leave_FunctionDef(
        self, _: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        return updated_node.with_changes(returns=None)


class _AssignHintRemover(cst.CSTTransformer):
    def leave_AnnAssign(
        self, _: cst.AnnAssign, updated_node: cst.AnnAssign
    ) -> cst.BaseSmallStatement | cst.RemovalSentinel:
        if updated_node.value is None:
            return cst.RemoveFromParent()

        return cst.Assign(targets=[cst.AssignTarget(updated_node.target)], value=updated_node.value)


class _HintRemover(_AssignHintRemover, _ParameterHintRemover, _ReturnHintRemover):
    pass


class TypeAnnotationApplierVisitor(codemod.ContextAwareTransformer):
    def __init__(
        self,
        context: codemod.CodemodContext,
        tycol: TypeCollection | pt.DataFrame[TypeCollectionSchema],
    ) -> None:
        super().__init__(context)
        self.tycol = tycol.df if isinstance(tycol, TypeCollection) else tycol

    def transform_module_impl(self, tree: cst.Module) -> cst.Module:
        assert self.context.filename is not None
        assert self.context.metadata_manager is not None

        relative = pathlib.Path(self.context.filename).relative_to(
            self.context.metadata_manager.root_path
        )

        module_tycol = self.tycol[self.tycol["file"] == str(relative)]
        #        req_mod_imports = module_tycol["anno"].str.split(".", n=1, regex=False, expand=True)
        #        if not req_mod_imports.empty:
        # viable_imports = req_mod_imports.set_axis(["pkg", "_"], axis=1)
        # viable_imports = viable_imports[
        # viable_imports["_"].notna() & viable_imports["pkg"].str.islower()
        # ].drop_duplicates(keep="first")
        # pkgs = viable_imports["pkg"].values
        #
        # for pkg in pkgs:
        # AddImportsVisitor.add_needed_import(self.context, pkg)

        removed = tree.visit(_HintRemover())
        annotations = TypeCollection.to_annotations(
            module_tycol.pipe(pt.DataFrame[TypeCollectionSchema])
        )

        with_ssa_qnames = FromQName2SSAQNameTransformer(
            context=self.context, annotations=annotations
        ).transform_module(removed)

        hinted = ApplyTypeAnnotationsVisitor(
            context=self.context,
            annotations=annotations,
            overwrite_existing_annotations=True,
            use_future_annotations=True,
            handle_function_bodies=True,
            create_class_attributes=True,
        ).transform_module(with_ssa_qnames)

        # ApplyTypeAnnotationsVisitor.store_stub_in_context(self.context, hinted)
        # imported = ApplyTypeAnnotationsVisitor(
        #    context=self.context, overwrite_existing_annotations=False
        # ).transform_module(hinted)

        with_qnames = FromSSAQName2QnameTransformer(
            context=self.context, annotations=annotations
        ).transform_module(hinted)

        return with_qnames


class ScopeAwareTransformer(codemod.ContextAwareTransformer):
    def __init__(self, context: codemod.CodemodContext) -> None:
        super().__init__(context)

        self._scope: list[tuple[str, ...]] = []

    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
        self._scope.append((*self.current_scope(), node.name.value))

    def leave_ClassDef(self, _: cst.ClassDef, updated: cst.ClassDef) -> cst.ClassDef:
        self._scope.pop()
        return updated

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        self._scope.append((*self.current_scope(), node.name.value))

    def leave_FunctionDef(self, _: cst.FunctionDef, updated: cst.FunctionDef) -> cst.FunctionDef:
        self._scope.pop()
        return updated

    def current_scope(self) -> tuple[str, ...]:
        return self._scope[-1] if self._scope else tuple()


class FromQName2SSAQNameTransformer(ScopeAwareTransformer):
    def __init__(
        self, context: codemod.CodemodContext, annotations: pt.DataFrame[TypeCollectionSchema]
    ) -> None:
        super().__init__(context)
        self.annotations = annotations.copy().assign(consumed=0)

    def transform_module_impl(self, tree: cst.Module) -> cst.Module:
        return self.visit_Module(tree)

    def leave_AnnAssign(self, _: cst.AnnAssign, updated_node: cst.AnnAssign) -> cst.AnnAssign:
        if (qname_ssa := self._handle_assn_tgt(updated_node.target)) is not None:
            return updated_node.with_changes(target=qname_ssa)
        return updated_node

    def leave_AssignTarget(
        self, _: cst.AssignTarget, updated_node: cst.AssignTarget
    ) -> cst.AssignTarget:
        if (qname_ssa := self._handle_assn_tgt(updated_node.target)) is not None:
            return updated_node.with_changes(target=qname_ssa)
        return updated_node

    def _handle_assn_tgt(
        self, node: cst.BaseAssignTargetExpression
    ) -> cst.BaseAssignTargetExpression | None:
        if m.matches(node, m.Name() | m.Attribute(value=m.Name("self"))):
            s, name = self.current_scope(), _stringify(node)
            assert name is not None
            full_qname = f"{'.'.join(s)}.{name}" if len(s) else name

            candidates = self.annotations.loc[
                (self.annotations[TypeCollectionSchema.qname] == full_qname)
                & (self.annotations["consumed"] != 1)
            ]
            candidates["consumed"].iloc[0] = 1
            qname_ssa = str(candidates[TypeCollectionSchema.qname_ssa].iloc[0])

            if isinstance(node, cst.Name):
                node.value = qname_ssa
            elif isinstance(node, cst.Attribute):
                node.value = cst.Name(qname_ssa)
            else:
                assert RuntimeError(f"Unexpected assign target type: {type(node)}")

            return node

        return None


class FromSSAQName2QnameTransformer(ScopeAwareTransformer):
    def __init__(
        self, context: codemod.CodemodContext, annotations: pt.DataFrame[TypeCollectionSchema]
    ) -> None:
        super().__init__(context)
        self.annotations = annotations

    def transform_module_impl(self, tree: cst.Module) -> cst.Module:
        return self.visit_Module(tree)

    def leave_AnnAssign(self, _: cst.AnnAssign, updated_node: cst.AnnAssign) -> cst.AnnAssign:
        if (qname_ssa := self._handle_assn_tgt(updated_node.target)) is not None:
            return updated_node.with_changes(target=qname_ssa)
        return updated_node

    def leave_AssignTarget(
        self, _: cst.AssignTarget, updated_node: cst.AssignTarget
    ) -> cst.AssignTarget:
        if (qname_ssa := self._handle_assn_tgt(updated_node.target)) is not None:
            return updated_node.with_changes(target=qname_ssa)
        return updated_node

    def _handle_assn_tgt(
        self, node: cst.BaseAssignTargetExpression
    ) -> cst.BaseAssignTargetExpression | None:
        if m.matches(node, m.Name() | m.Attribute(value=m.Name("self"))):
            s, name = self.current_scope(), _stringify(node)
            assert name is not None
            full_qname = f"{'.'.join(s)}.{name}" if len(s) else name

            candidates = self.annotations.loc[
                (self.annotations[TypeCollectionSchema.qname_ssa] == full_qname)
            ]
            qname = str(candidates[TypeCollectionSchema.qname].iloc[0])

            if isinstance(node, cst.Name):
                node.value = qname
            elif isinstance(node, cst.Attribute):
                node.value = cst.Name(qname)
            else:
                assert RuntimeError(f"Unexpected assign target type: {type(node)}")

            return node

        return None
