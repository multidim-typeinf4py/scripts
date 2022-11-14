import difflib
import pathlib

import click
import libcst as cst
from libcst.codemod import _cli as cstcli


@click.command(name="diff", short_help="test based unified diff of the provided files")
@click.option(
    "-i",
    "--inputs",
    nargs=2,
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path
    ),
    required=True,
    help="Files to diff"
)
def cli(inputs: tuple[pathlib.Path, pathlib.Path]) -> None:
    former, latter = inputs
    former_module, latter_module = (
        cst.parse_module(former.open().read()),
        cst.parse_module(latter.open().read()),
    )

    diff = cstcli.diff_code(former_module.code, latter_module.code)
    print(diff)

if __name__ == "__main__":
    cli()