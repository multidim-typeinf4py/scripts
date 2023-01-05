import functools
import pathlib

from libcst import codemod
import libcst as cst
from libcst.codemod.visitors._apply_type_annotations import ApplyTypeAnnotationsVisitor
from libcst import codemod

import pandera.typing as pt

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


class TypeAnnotationApplierVisitor(codemod.Codemod):
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

        transformers = [
            _HintRemover(),
            ApplyTypeAnnotationsVisitor(
                context=self.context,
                annotations=TypeCollection.to_annotations(
                    module_tycol.pipe(pt.DataFrame[TypeCollectionSchema])
                ),
                overwrite_existing_annotations=True,
                use_future_annotations=True,
                handle_function_bodies=True,
                create_class_attributes=True,
            ),
        ]

        # importer = AddImportsVisitor(context=self.context)
        # tree = tree.visit(importer)

        return functools.reduce(lambda m, t: m.visit(t), transformers, tree)
