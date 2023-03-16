import pathlib

import libcst
import pandera.typing as pt
from libcst import codemod

from common.annotations import ApplyTypeAnnotationsVisitor
from common.schemas import TypeCollectionSchema
from common.storage import TypeCollection

from .qname_transforms import QName2SSATransformer, SSA2QNameTransformer
from .lower_transforms import LoweringTransformer, UnloweringTransformer

from symbols.collector import TypeCollectorVisitor


class TypeAnnotationApplierTransformer(codemod.ContextAwareTransformer):
    def __init__(
        self,
        context: codemod.CodemodContext,
        annotations: pt.DataFrame[TypeCollectionSchema],
    ) -> None:
        super().__init__(context)
        self.annotations = annotations

    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        assert self.context.filename is not None
        assert self.context.metadata_manager is not None

        relative = pathlib.Path(self.context.filename).relative_to(
            self.context.metadata_manager.root_path
        )

        module_tycol = self.annotations[
            self.annotations[TypeCollectionSchema.file] == str(relative)
        ].pipe(pt.DataFrame[TypeCollectionSchema])

        # AddImportsVisitor.add_needed_import(self.context, "typing")
        # AddImportsVisitor.add_needed_import(self.context, "typing", "*")

        # removed = tree.visit(TypeAnnotationRemover(context=self.context))

        symbol_collector = TypeCollectorVisitor.strict(context=self.context)
        tree.visit(symbol_collector)
        annotations = TypeCollection.to_libcst_annotations(
            module_tycol, symbol_collector.collection.df
        )

        lowered = LoweringTransformer(context=self.context).transform_module(tree)

        with_ssa_qnames = QName2SSATransformer(
            context=self.context, annotations=module_tycol
        ).transform_module(lowered)

        hinted = ApplyTypeAnnotationsVisitor(
            context=self.context,
            annotations=annotations,
            overwrite_existing_annotations=False,
            use_future_annotations=True,
        ).transform_module(with_ssa_qnames)

        with_qnames = SSA2QNameTransformer(
            context=self.context, annotations=module_tycol
        ).transform_module(hinted)

        unlowered = UnloweringTransformer(context=self.context).transform_module(
            with_qnames
        )
        return unlowered
