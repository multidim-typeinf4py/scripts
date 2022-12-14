import pandera.typing as pt

from pyre_check.client import command_arguments, commands, configuration

from . import _adaptors

from ._base import ProjectWideInference
from common.schemas import TypeCollectionSchema


class Pyre(ProjectWideInference):
    method = "pyre"

    def _pyre_infer(self) -> None:
        config = configuration.create_configuration(
            arguments=command_arguments.CommandArguments(), base_directory=self.project
        )
        infargs = command_arguments.InferArguments(
            working_directory=self.project,
            annotate_attributes=True,
            annotate_from_existing_stubs=False,
            debug_infer=False,
            quote_annotations=True,
            dequalify=False,
            in_place=False,
            print_only=False,
            read_stdin=False,
        )

        assert (
            commands.infer.run(configuration=config, query_arguments=infargs)
            != commands.ExitCode.FAILURE
        )

    def _infer_project(self) -> pt.DataFrame[TypeCollectionSchema]:
        self._pyre_infer()
        return _adaptors.stubs2df(stub_folder=self.project / ".pyre" / "types")
