from __future__ import annotations
import logging

import pathlib

import libcst as cst

import libcst.codemod as codemod
import libcst.metadata as metadata

from common import TypeCollection


# TODO: technically not accurate as this is a visitor, not a transformer
# TODO: but there does not seem to be a nicer way to execute this visitor in parallel
class TypeCollectorVistor(codemod.ContextAwareTransformer):
    collection: TypeCollection

    def __init__(
        self, context: codemod.CodemodContext, collection: TypeCollection, strict: bool
    ) -> None:
        super().__init__(context)
        self.collection = collection
        self.strict = strict
        self.logger = logging.getLogger(self.__class__.__qualname__)

    @staticmethod
    def strict(context: codemod.CodemodContext) -> TypeCollectorVistor:
        return TypeCollectorVistor(context=context, collection=TypeCollection.empty(), strict=True)

    @staticmethod
    def lax(context: codemod.CodemodContext) -> TypeCollectorVistor:
        return TypeCollectorVistor(context=context, collection=TypeCollection.empty(), strict=False)

    def transform_module_impl(self, tree: cst.Module) -> cst.Module:
        assert self.context.filename is not None
        assert self.context.metadata_manager is not None

        file = pathlib.Path(self.context.filename).relative_to(
            self.context.metadata_manager.root_path
        )
        self.logger.info(f"Collecting from {file}")

        from libcst.codemod.visitors._apply_type_annotations import (
            TypeCollector as LibCSTTypeCollector,
        )
        from libcst.codemod.visitors._gather_imports import (
            GatherImportsVisitor,
        )

        metadataed = metadata.MetadataWrapper(tree)

        imports_visitor = GatherImportsVisitor(context=self.context)
        metadataed.visit(imports_visitor)

        existing_imports = set(
            item.module for item in imports_visitor.symbol_mapping.values()
        )

        type_collector = LibCSTTypeCollector(
            existing_imports=existing_imports,
            module_imports=imports_visitor.symbol_mapping,
            context=self.context,
            create_class_attributes=True,
            handle_function_bodies=True,
            track_unannotated=True,
        )

        metadataed.visit(type_collector)
        update = TypeCollection.from_annotations(
            file=file, annotations=type_collector.annotations, strict=self.strict
        )

        self.collection.merge(update)
        return tree
