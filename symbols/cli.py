import pathlib
import sys

import click

from libcst.codemod import _cli as cstcli
import libcst.codemod as codemod

from .collector import TypeCollectorVistor


@click.command(
    name="symbols",
    short_help="Traverse the given repository and collect annotated tokens",
)
@click.option(
    "-i",
    "--input",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path
    ),
    required=True,
    help="Repository to gather",
)
def entrypoint(root: pathlib.Path) -> None:
    result = codemod.parallel_exec_transform_with_prettyprint(
        transform=TypeCollectorVistor.initial(context=codemod.CodemodContext()),
        files=cstcli.gather_files(str(root)),
        jobs=1,
        blacklist_patterns=["__init__.py"],
        repo_root=str(root),
    )

    print(
        f"Finished codemodding {result.successes + result.skips + result.failures} files!",
        file=sys.stderr,
    )
    print(
        f" - Collected symbol from {result.successes} files successfully.",
        file=sys.stderr,
    )
    print(f" - Skipped {result.skips} files.", file=sys.stderr)
    print(f" - Failed to collect from {result.failures} files.", file=sys.stderr)
    print(f" - {result.warnings} warnings were generated.", file=sys.stderr)


if __name__ == "__main__":
    entrypoint()
