from __future__ import annotations

import logging
import pathlib
from dataclasses import replace

import libcst as cst
import tqdm
from libcst import codemod as codemod, metadata

from common import TypeCollection


def build_type_collection(root: pathlib.Path, allow_stubs=False) -> TypeCollection:
    repo_root = str(root.parent if root.is_file() else root)
    files = (
        [str(root)]
        if root.is_file()
        else codemod.gather_files([str(root)], include_stubs=allow_stubs)
    )

    visitor = TypeCollectorVisitor.strict(
        context=codemod.CodemodContext(
            metadata_manager=metadata.FullRepoManager(
                repo_root_dir=repo_root,
                paths=files,
                providers=[],
            )
        )
    )

    for file in (pbar := tqdm.tqdm(files)):
        pbar.set_description(f"Building Annotation Collection from {file}")
        visitor.context = replace(visitor.context, filename=file)

        module = cst.parse_module(open(file).read())
        module.visit(visitor)

    return visitor.collection


class TypeCollectorVisitor(codemod.ContextAwareVisitor):
    collection: TypeCollection

    def __init__(
        self, context: codemod.CodemodContext, collection: TypeCollection, strict: bool
    ) -> None:
        super().__init__(context)
        self.collection = collection
        self._strict = strict
        self.logger = logging.getLogger(self.__class__.__qualname__)

    @staticmethod
    def strict(context: codemod.CodemodContext) -> TypeCollectorVisitor:
        return TypeCollectorVisitor(context=context, collection=TypeCollection.empty(), strict=True)

    @staticmethod
    def lax(context: codemod.CodemodContext) -> TypeCollectorVisitor:
        return TypeCollectorVisitor(context=context, collection=TypeCollection.empty(), strict=False)

    def visit_Module(self, tree: cst.Module) -> None:
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

        # metadataed = metadata.MetadataWrapper(tree)

        imports_visitor = GatherImportsVisitor(context=self.context)
        tree.visit(imports_visitor)

        existing_imports = set(item.module for item in imports_visitor.symbol_mapping.values())

        type_collector = MultiVarTypeCollector(
            existing_imports=existing_imports,
            module_imports=imports_visitor.symbol_mapping,
            context=self.context,
        )

        metadata.MetadataWrapper(tree).visit(type_collector)
        update = TypeCollection.from_annotations(
            file=file, annotations=type_collector.annotations, strict=self._strict
        )

        self.collection.merge_into(update)
