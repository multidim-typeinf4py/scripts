import pandera.typing as pt

from pyre_check.client import command_arguments, commands, configuration

from . import _adaptors

from ._base import ProjectWideInference, scratchpad
from common.schemas import TypeCollectionSchema


# TODO: Consider libcst's TypeInferenceProvider instead, which uses Pyre anyhow

class Pyre(ProjectWideInference):
    method = "pyre"

    _OUTPUT_DIR = ".pyre-stubs"

    def _infer_project(self) -> pt.DataFrame[TypeCollectionSchema]:
        with scratchpad(self.project) as sp:
            config = configuration.create_configuration(
                arguments=command_arguments.CommandArguments(
                    dot_pyre_directory=sp / Pyre._OUTPUT_DIR
                ),
                base_directory=sp,
            )
            infargs = command_arguments.InferArguments(
                working_directory=sp,
                annotate_attributes=True,
                annotate_from_existing_stubs=True,
                debug_infer=False,
                quote_annotations=False,
                dequalify=False,
                in_place=True,
                print_only=False,
                read_stdin=False,
            )

            assert (
                commands.infer.run(configuration=config, query_arguments=infargs)
                != commands.ExitCode.FAILURE
            )

            return _adaptors.hints2df(sp)
