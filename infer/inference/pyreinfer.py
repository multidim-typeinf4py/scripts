import pandera.typing as pt
from pyre_check.client import command_arguments, commands, configuration

from common.schemas import InferredSchema
from utils import working_dir
from . import _adaptors
from ._base import ProjectWideInference


class PyreInfer(ProjectWideInference):
    method = "pyre-infer"

    _OUTPUT_DIR = ".pyre-stubs"

    def _infer_project(self) -> pt.DataFrame[InferredSchema]:
        with working_dir(self.mutable):
            config = configuration.create_configuration(
                arguments=command_arguments.CommandArguments(
                    dot_pyre_directory=self.mutable / PyreInfer._OUTPUT_DIR,
                    source_directories=[str(self.mutable)],
                ),
                base_directory=self.mutable,
            )
            infargs = command_arguments.InferArguments(
                working_directory=self.mutable,
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
                _adaptors.hints2df(self.mutable)
                .assign(method=self.method, topn=1)
                .pipe(pt.DataFrame[InferredSchema])
            )
