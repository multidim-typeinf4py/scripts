import pandera.typing as pt

from pyre_check.client import command_arguments, commands, configuration

from . import _adaptors

from ._base import ProjectWideInference, scratchpad
from common.schemas import TypeCollectionSchema


class Pyre(ProjectWideInference):
    method = "pyre"

    def _infer_project(self) -> pt.DataFrame[TypeCollectionSchema]:
        with scratchpad(self.project) as sp:
            config = configuration.create_configuration(
                arguments=command_arguments.CommandArguments(),
                base_directory=sp,
            )
            infargs = command_arguments.InferArguments(
                working_directory=sp,
                annotate_attributes=True,
                annotate_from_existing_stubs=True,
                debug_infer=False,
                quote_annotations=True,
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