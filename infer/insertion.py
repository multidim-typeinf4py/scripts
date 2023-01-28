import logging
import pathlib

from libcst import codemod
import libcst as cst
from libcst.codemod.visitors._add_imports import AddImportsVisitor
from libcst.codemod.visitors._apply_type_annotations import ApplyTypeAnnotationsVisitor
import libcst.matchers as m

import pandera.typing as pt

from common.ast_helper import _stringify
from common.schemas import TypeCollectionSchema
from common.storage import TypeCollection
from symbols.collector import TypeCollectorVistor

from infer.removal import HintRemover


class TypeAnnotationApplierTransformer(codemod.ContextAwareTransformer):
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

        module_tycol = self.tycol[self.tycol[TypeCollectionSchema.file] == str(relative)].pipe(
            pt.DataFrame[TypeCollectionSchema]
        )

        AddImportsVisitor.add_needed_import(self.context, "typing")
        AddImportsVisitor.add_needed_import(self.context, "typing", "*")

        removed = tree.visit(HintRemover(self.context))

        symbol_collector = TypeCollectorVistor.strict(context=self.context)
        removed = symbol_collector.transform_module(removed)

        with_ssa_qnames = QName2SSATransformer(
            context=self.context, annotations=module_tycol
        ).transform_module(removed)

        annotations = TypeCollection.to_libcst_annotations(
            module_tycol, symbol_collector.collection.df
        )

        # Due to renaming, it it safe to use LibCST's implementation for this!
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

        with_qnames = SSA2QNameTransformer(
            context=self.context, annotations=module_tycol
        ).transform_module(hinted)

        return with_qnames


class ScopeAwareTransformer(codemod.ContextAwareTransformer):
    def __init__(self, context: codemod.CodemodContext) -> None:
        super().__init__(context)
        self._scope: list[tuple[str, ...]] = []

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self._scope.append((*self.current_scope(), node.name.value))

    def leave_ClassDef(self, _: cst.ClassDef, updated: cst.ClassDef) -> cst.ClassDef:
        self._scope.pop()
        return updated

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self._scope.append((*self.current_scope(), node.name.value))

    def leave_FunctionDef(self, _: cst.FunctionDef, updated: cst.FunctionDef) -> cst.FunctionDef:
        self._scope.pop()
        return updated

    def current_scope(self) -> tuple[str, ...]:
        return self._scope[-1] if self._scope else tuple()


class QName2SSATransformer(ScopeAwareTransformer):
    def __init__(
        self, context: codemod.CodemodContext, annotations: pt.DataFrame[TypeCollectionSchema]
    ) -> None:
        super().__init__(context)
        self.annotations = annotations.copy().assign(consumed=0)
        self.logger = logging.getLogger(QName2SSATransformer.__qualname__)

    def leave_Module(self, _: cst.Module, updated_node: cst.Module) -> cst.Module:
        diff_qnames = (
            self.annotations[TypeCollectionSchema.qname]
            != self.annotations[TypeCollectionSchema.qname_ssa]
        )
        unconsumed = self.annotations["consumed"] != 1
        flagged = self.annotations[diff_qnames & unconsumed]

        if not flagged.empty:
            self.logger.warning(
                f"Failed to apply qname_ssas for {flagged[[TypeCollectionSchema.qname, TypeCollectionSchema.qname_ssa]].values}"
            )

        return updated_node

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
            s, name = ".".join(self.current_scope()), _stringify(node)
            assert name is not None
            full_qname = f"{s}.{name}" if s else name

            cand_mask = (self.annotations[TypeCollectionSchema.qname] == full_qname) & (
                self.annotations["consumed"] != 1
            )
            candidates = self.annotations.loc[cand_mask]
            if candidates.empty:
                # self.logger.warning(f"Could not lookup {full_qname}")
                return

            qname_ssa_ser = candidates[TypeCollectionSchema.qname_ssa]
            qname_ssa = str(qname_ssa_ser.iloc[0])
            if s:
                qname_ssa = qname_ssa.removeprefix(s + ".")
            self.annotations.at[qname_ssa_ser.index[0], "consumed"] = 1

            e = cst.parse_expression(qname_ssa)
            assert isinstance(e, cst.BaseAssignTargetExpression)

            return e

        return None


class SSA2QNameTransformer(ScopeAwareTransformer):
    def __init__(
        self, context: codemod.CodemodContext, annotations: pt.DataFrame[TypeCollectionSchema]
    ) -> None:
        super().__init__(context)
        self.annotations = annotations.copy().assign(consumed=0)
        self.logger = logging.getLogger(SSA2QNameTransformer.__qualname__)

    def leave_Module(self, _: cst.Module, updated_node: cst.Module) -> cst.Module:
        diff_qnames = (
            self.annotations[TypeCollectionSchema.qname]
            != self.annotations[TypeCollectionSchema.qname_ssa]
        )
        unconsumed = self.annotations["consumed"] != 1
        flagged = self.annotations[diff_qnames & unconsumed]

        if not flagged.empty:
            self.logger.warning(
                f"Failed to apply qname_ssas for {flagged[[TypeCollectionSchema.qname_ssa, TypeCollectionSchema.qname]].values}"
            )

        return updated_node

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
            s, name = ".".join(self.current_scope()), _stringify(node)
            assert name is not None
            full_qname_ssa = f"{s}.{name}" if s else name

            cand_mask = self.annotations[TypeCollectionSchema.qname_ssa] == full_qname_ssa
            candidates = self.annotations.loc[cand_mask]
            if candidates.empty:
                # self.logger.warning(f"Could not lookup {full_qname_ssa}")
                return

            qname_ser = candidates[TypeCollectionSchema.qname]
            qname = qname_ser.iloc[0]
            assert (
                candidates["consumed"].iloc[0] == 0
            ), f"Attempted to reapply {full_qname_ssa} -> {qname} more than once!"
            self.annotations.at[qname_ser.index[0], "consumed"] = 1

            if s:
                qname = qname.removeprefix(s + ".")
            e = cst.parse_expression(qname)
            assert isinstance(e, cst.BaseAssignTargetExpression)

            return e

        return None
