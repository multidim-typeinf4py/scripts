from __future__ import annotations
import logging

import pathlib

import libcst as cst

import libcst.codemod as codemod
import libcst.metadata as metadata

from common import TypeCollection


def build_type_collection(root: pathlib.Path, allow_stubs=False) -> TypeCollection:
    repo_root = str(root.parent if root.is_file() else root)
    files = (
        [str(root)]
        if root.is_file()
        else codemod.gather_files([str(root)], include_stubs=allow_stubs)
    )

    visitor = TypeCollectorVistor.strict(context=codemod.CodemodContext())
    _ = codemod.parallel_exec_transform_with_prettyprint(
        transform=visitor,
        files=files,
        jobs=1,
        repo_root=repo_root,
    )

    return visitor.collection


# TODO: technically not accurate as this is a visitor, not a transformer
# TODO: but there does not seem to be a nicer way to get context auto-injected
class TypeCollectorVistor(codemod.ContextAwareTransformer):
    collection: TypeCollection

    def __init__(
        self, context: codemod.CodemodContext, collection: TypeCollection, strict: bool
    ) -> None:
        super().__init__(context)
        self.collection = collection
        self._strict = strict
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

        from common.annotations import MultiVarTypeCollector

        from libcst.codemod.visitors._gather_imports import (
            GatherImportsVisitor,
        )

        metadataed = metadata.MetadataWrapper(tree)

        imports_visitor = GatherImportsVisitor(context=self.context)
        metadataed.visit(imports_visitor)

        existing_imports = set(item.module for item in imports_visitor.symbol_mapping.values())

        type_collector = MultiVarTypeCollector(
            existing_imports=existing_imports,
            module_imports=imports_visitor.symbol_mapping,
            context=self.context,
        )

        metadataed.visit(type_collector)
        update = TypeCollection.from_annotations(
            file=file, annotations=type_collector.annotations, strict=self._strict
        )

        self.collection.merge_into(update)
        return tree
