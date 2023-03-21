import pandera.typing as pt

from pyre_check.client import command_arguments, commands, configuration

from . import _adaptors

from ._base import ProjectWideInference
from utils import scratchpad, working_dir
from common.schemas import InferredSchema



class PyreInfer(ProjectWideInference):
    method = "pyre-infer"

    _OUTPUT_DIR = ".pyre-stubs"

    def _infer_project(self) -> pt.DataFrame[InferredSchema]:
        with scratchpad(self.project) as sp, working_dir(sp):
            config = configuration.create_configuration(
                arguments=command_arguments.CommandArguments(
                    dot_pyre_directory=sp / PyreInfer._OUTPUT_DIR, source_directories=[str(sp)]
                ),
                base_directory=sp,
            )
            infargs = command_arguments.InferArguments(
                working_directory=sp,
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
                _adaptors.stubs2df(sp / PyreInfer._OUTPUT_DIR / "types")
                .assign(method=self.method, topn=1)
                .pipe(pt.DataFrame[InferredSchema])
            )
