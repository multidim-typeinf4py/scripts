import logging
import pathlib
from typing import Optional

import libcst
import pandera.typing as pt
from libcst import codemod
from libcst.codemod import visitors

import utils
from src.common.schemas import TypeCollectionSchema
from src.symbols.collector import build_type_collection


def hints2df(
    folder: pathlib.Path, subset: Optional[set[pathlib.Path]]
) -> pt.DataFrame[TypeCollectionSchema]:
    collection = build_type_collection(folder, allow_stubs=False, subset=subset)
    return collection.df


class ParallelStubber(codemod.ContextAwareTransformer):
    def __init__(
        self,
        context: codemod.CodemodContext,
        project_folder: pathlib.Path,
        stub_folder: pathlib.Path,
    ) -> None:
        super().__init__(context)
        self.project_folder = project_folder
        self.stub_folder = stub_folder

        self.logger = logging.getLogger(type(self).__qualname__)

    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        assert self.context.filename is not None
        rel_stub_path = (
            pathlib.Path(self.context.filename)
            .relative_to(self.project_folder)
            .with_suffix(".pyi")
        )
        stubfile = self.stub_folder / rel_stub_path
        if not stubfile.is_file():
            self.logger.warning(
                f"Could not find {rel_stub_path} in {self.stub_folder}; {self.context.filename} remains unchanged"
            )
            return tree

        visitor = visitors.ApplyTypeAnnotationsVisitor
        visitor.store_stub_in_context(
            context=self.context, stub=libcst.parse_module(stubfile.read_text())
        )

        return visitor(
            context=self.context,
            overwrite_existing_annotations=True,
            use_future_annotations=True,
            strict_posargs_matching=False,
            strict_annotation_matching=True,
        ).transform_module(tree)


def stubs2df(
    project_folder: pathlib.Path,
    stubs_folder: pathlib.Path,
    subset: set[pathlib.Path],
) -> pt.DataFrame[TypeCollectionSchema]:
    files = list(map(lambda p: str(project_folder / p), subset))
    stubbing_result = codemod.parallel_exec_transform_with_prettyprint(
        transform=ParallelStubber(
            context=codemod.CodemodContext(),
            project_folder=project_folder,
            stub_folder=stubs_folder,
        ),
        jobs=utils.worker_count(),
        repo_root=str(project_folder),
        files=files,
    )
    print(utils.format_parallel_exec_result("Applying Stubs", result=stubbing_result))

    collection = build_type_collection(project_folder, allow_stubs=False, subset=subset)
    df = collection.df

    return df
