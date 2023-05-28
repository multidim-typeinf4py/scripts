import pathlib
from typing import Optional

from libsa4py import pyre
import pandera.typing as pt
from pyre_check.client import command_arguments, commands, configuration

from common.schemas import InferredSchema
from utils import working_dir
from . import _adaptors
from ._base import ProjectWideInference


class PyreInfer(ProjectWideInference):
    def method(self) -> str:
        return "pyre-infer"

    _OUTPUT_DIR = ".pyre-stubs"

    def _infer_project(
        self, mutable: pathlib.Path, subset: set[pathlib.Path]
    ) -> pt.DataFrame[InferredSchema]:
        with working_dir(mutable):
            pyre.clean_pyre_config(project_path=str(mutable))
            config = configuration.create_configuration(
                arguments=command_arguments.CommandArguments(
                    dot_pyre_directory=mutable / PyreInfer._OUTPUT_DIR,
                    source_directories=[str(mutable)],
                ),
                base_directory=mutable,
            )
            commands.initialize.write_configuration(
                configuration=dict(
                    site_package_search_strategy="pep561",
                    source_directories=["."],
                ),
                configuration_path=mutable / ".pyre_configuration",
            )

            infargs = command_arguments.InferArguments(
                working_directory=mutable,
                annotate_attributes=True,
                annotate_from_existing_stubs=False,
                debug_infer=False,
                quote_annotations=False,
                dequalify=False,
                in_place=False,
                print_only=False,
                read_stdin=False,
            )

            try:
                exitcode = commands.infer.run(configuration=config, infer_arguments=infargs)

            except commands.ClientException:
                self.logger.error("pyre-infer encountered an internal failure, cannot infer types")
                return InferredSchema.example(size=0)

            if exitcode != commands.ExitCode.SUCCESS:
                self.logger.warning(f"pyre-infer indicated {exitcode} instead of {commands.ExitCode.SUCCESS}; proceed with caution")

            return (
                _adaptors.stubs2df(mutable / PyreInfer._OUTPUT_DIR / "types", subset=subset)
                .assign(method=self.method(), topn=1)
                .pipe(pt.DataFrame[InferredSchema])
            )
