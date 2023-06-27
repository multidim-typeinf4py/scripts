import dataclasses
import pathlib

import libcst
import pandera.typing as pt
from libcst import codemod, metadata

from scripts.common.annotations import ApplyTypeAnnotationsVisitor
from scripts.common.schemas import TypeCollectionSchema
from scripts.common.storage import TypeCollection

from .qname_transforms import QName2SSATransformer, SSA2QNameTransformer

from scripts.symbols.collector import TypeCollectorVisitor


class TypeAnnotationApplierTransformer(codemod.ContextAwareTransformer):
    def __init__(
        self,
        context: codemod.CodemodContext,
        annotations: pt.DataFrame[TypeCollectionSchema],
    ) -> None:
        super().__init__(context)
        self.annotations = annotations

        self.annotations[TypeCollectionSchema.anno] = self.annotations[
            TypeCollectionSchema.anno
        ].str.removeprefix("builtins.")

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

        metadata_manager = metadata.FullRepoManager(
            repo_root_dir=self.context.metadata_manager.root_path,
            paths=[self.context.filename],
            providers={metadata.FullyQualifiedNameProvider},
        )
        metadata_manager.resolve_cache()

        symbol_collector = TypeCollectorVisitor.strict(
            context=dataclasses.replace(self.context, metadata_manager=metadata_manager)
        )
        tree.visit(symbol_collector)
        annotations = TypeCollection.to_libcst_annotations(
            module_tycol, symbol_collector.collection.df
        )

        # lowered = LoweringTransformer(context=self.context).transform_module(tree)

        with_ssa_qnames = QName2SSATransformer(
            context=self.context, annotations=module_tycol
        ).transform_module(tree)

        hinted = ApplyTypeAnnotationsVisitor(
            context=self.context,
            annotations=annotations,
            overwrite_existing_annotations=False,
            use_future_annotations=True,
        ).transform_module(with_ssa_qnames)

        with_qnames = SSA2QNameTransformer(
            context=self.context, annotations=module_tycol
        ).transform_module(hinted)

        # unlowered = UnloweringTransformer(context=self.context).transform_module(
        #    with_qnames
        # )
        return with_qnames
