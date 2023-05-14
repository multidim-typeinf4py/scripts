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
    method = "pyre-infer"

    _OUTPUT_DIR = ".pyre-stubs"

    def _infer_project(
        self, mutable: pathlib.Path, subset: Optional[set[pathlib.Path]]
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
                in_place=True,
                print_only=False,
                read_stdin=False,
            )

            assert (
                commands.infer.run(configuration=config, infer_arguments=infargs)
                != commands.ExitCode.FAILURE
            )

            return (
                _adaptors.stubs2df(mutable / PyreInfer._OUTPUT_DIR / "types", subset=subset)
                .assign(method=self.method, topn=1)
                .pipe(pt.DataFrame[InferredSchema])
            )
