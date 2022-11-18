import pathlib
import sys

import click

from libcst.codemod import _cli as cstcli
import libcst.codemod as codemod

from common import TypeCollection
from .collector import TypeCollectorVistor


@click.command(
    name="symbols",
    short_help="Traverse the given repository and collect annotated tokens",
)
@click.option(
    "-r",
    "--root",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path
    ),
    required=True,
    help="Root of Repository to gather annotations from",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(
        exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path
    ),
    required=True,
    help="Output path for .tsv",
)
def entrypoint(root: pathlib.Path, output: pathlib.Path) -> None:
    result, collection = _collect(root)
    if result.failures or result.warnings:
        print(
            "WARNING: Failures and / or warnings occurred, the symbol collection may be incomplete!"
        )
    _store(collection, output)


def _collect(
    root: pathlib.Path,
) -> tuple[cstcli.ParallelTransformResult, TypeCollection]:
    repo_root = str(root.parent if root.is_file() else root)

    visitor = TypeCollectorVistor.strict(context=codemod.CodemodContext())
    result = codemod.parallel_exec_transform_with_prettyprint(
        transform=visitor,
        files=cstcli.gather_files([str(root)]),
        jobs=1,
        blacklist_patterns=["__init__.py"],
        repo_root=repo_root,
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

    return result, visitor.collection


def _store(collection: TypeCollection, output: pathlib.Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    collection.write(output)


if __name__ == "__main__":
    entrypoint()
