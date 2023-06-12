import contextlib
import functools
import pathlib
import shutil

import libcst
import pandera.typing as pt
from libcst import codemod
from libsa4py import pyre
from pyre_check.client.command_arguments import CommandArguments, StartArguments
from pyre_check.client.commands import initialize, start, ExitCode, stop
from pyre_check.client.configuration import configuration
from pyre_check.client.identifiers import PyreFlavor

from scripts import utils
from scripts.common.schemas import InferredSchema
from scripts.infer.annotators.pyrequery import (
    PyreQueryFileApplier,
)
from scripts.infer.annotators.tool_annotator import Normalisation
from scripts.symbols.collector import build_type_collection
from ._base import ProjectWideInference


class NormalisedPyreQuery(codemod.Codemod):
    NORMALISER = Normalisation(
        bad_list_generics=True,
        bad_tuple_generics=True,
        bad_dict_generics=True,
        lowercase_aliases=True,
        unnest_union_t=True,
        typing_text_to_str=True,
    )

    def __init__(self, context: codemod.CodemodContext) -> None:
        super().__init__(context)

    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        inferred = PyreQueryFileApplier(context=self.context).transform_module(tree)
        normalised = functools.reduce(
            lambda mod, transformer: transformer.transform_module(mod),
            NormalisedPyreQuery.NORMALISER.transformers(self.context),
            inferred,
        )

        return normalised

@contextlib.contextmanager
def pyre_server(project_location: pathlib.Path) -> None:
    pyre.clean_pyre_config(project_path=str(project_location))

    src_dirs = [str(project_location)]
    cmd_args = CommandArguments(source_directories=src_dirs)

    config = configuration.create_configuration(
        arguments=cmd_args,
        base_directory=project_location,
    )

    initialize.write_configuration(
        configuration=dict(
            site_package_search_strategy="pep561",
            source_directories=["."],
        ),
        configuration_path=project_location / ".pyre_configuration",
    )

    start_args = StartArguments.create(
        command_argument=cmd_args, no_watchman=True,
    )
    assert start.run(configuration=config, start_arguments=start_args) == ExitCode.SUCCESS

    yield

    assert stop.run(config, flavor=PyreFlavor.CLASSIC) == ExitCode.SUCCESS
    if (dotdir := project_location / ".pyre").is_dir():
        shutil.rmtree(str(dotdir))


class PyreQuery(ProjectWideInference):
    def method(self) -> str:
        return "pyrequery"

    def _infer_project(
        self, mutable: pathlib.Path, subset: set[pathlib.Path]
    ) -> pt.DataFrame[InferredSchema]:
        with utils.working_dir(wd=mutable), pyre_server(mutable):
            files = [str(mutable / s) for s in subset]

            # Execute serially to avoid potential race-conditions
            annotated = codemod.parallel_exec_transform_with_prettyprint(
                transform=PyreQueryFileApplier(context=codemod.CodemodContext()),
                jobs=1,
                repo_root=str(mutable),
                files=files,
            )

            anno_res = utils.format_parallel_exec_result(
                f"Annotated with {self.method()}", result=annotated
            )
            self.logger.info(anno_res)

            collected = build_type_collection(
                root=mutable,
                allow_stubs=False,
                subset=subset
            ).df
        return collected.assign(method=self.method(), topn=1).pipe(
            pt.DataFrame[InferredSchema]
        )
