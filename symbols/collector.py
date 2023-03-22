from __future__ import annotations

import logging
import pathlib

import libcst as cst
from libcst import codemod as c, metadata
import tqdm
from tqdm.contrib.concurrent import process_map

import pandas as pd
from pandera import typing as pt

from common import TypeCollection
from common.schemas import TypeCollectionSchema


class _ParallelTypeCollector:
    def __init__(self, repo_root: str, files: list[str]) -> None:
        self.repo_root = repo_root
        self.files = files

    def __call__(self, filename2code: tuple[str, str]) -> pt.DataFrame[TypeCollectionSchema]:
        file, code = filename2code
        context = c.CodemodContext(
            filename=file,
            metadata_manager=metadata.FullRepoManager(
                repo_root_dir=self.repo_root,
                paths=self.files,
                providers=[],
            ),
        )
        visitor = TypeCollectorVisitor.strict(context=context)

        try:
            module = cst.parse_module(code)
            module.visit(visitor)
        except Exception as e:
            print(f"FILE {file}: EXCEPTION OCCURRED: {e}")
            return TypeCollectionSchema.example(size=0)

        return visitor.collection.df


def build_type_collection(root: pathlib.Path, allow_stubs=False) -> TypeCollection:
    repo_root = str(root.parent if root.is_file() else root)
    files = (
        [str(root)] if root.is_file() else c.gather_files([str(root)], include_stubs=allow_stubs)
    )

    file2code = {file: open(file).read() for file in tqdm.tqdm(files)}

    collector = _ParallelTypeCollector(repo_root=repo_root, files=files)
    collections = process_map(
        collector,
        file2code.items(),
        total=len(file2code),
        desc=f"Building Type Collection from {root}",
    )
    return TypeCollection(pd.concat(collections, ignore_index=True))


class TypeCollectorVisitor(c.ContextAwareVisitor):
    collection: TypeCollection

    def __init__(self, context: c.CodemodContext, collection: TypeCollection, strict: bool) -> None:
        super().__init__(context)
        self.collection = collection
        self._strict = strict
        self.logger = logging.getLogger(self.__class__.__qualname__)

    @staticmethod
    def strict(context: c.CodemodContext) -> TypeCollectorVisitor:
        return TypeCollectorVisitor(context=context, collection=TypeCollection.empty(), strict=True)

    @staticmethod
    def lax(context: c.CodemodContext) -> TypeCollectorVisitor:
        return TypeCollectorVisitor(
            context=context, collection=TypeCollection.empty(), strict=False
        )

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
