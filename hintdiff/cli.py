import os
import pathlib
import sys
import typing

import click
from libcst.codemod import _cli as cstcli
import libcst.codemod as codemod
import pandas as pd


from common.storage import MergedAnnotations
from symbols.collector import TypeCollectorVistor

from . import coverage, hintstat


@click.command(
    name="hintdiff",
    short_help="Compare symbols and their annotations across provided repositories",
)
@click.option(
    "-r",
    "--repo",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path),
    multiple=True,
    required=True,
    help="Repositories to collect from",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    required=True,
    help="Output path for .tsv and derived statistics",
)
@click.option(
    "-s",
    "--statistic",
    type=click.Choice(choices=list(hintstat.Statistic.__members__.keys()), case_sensitive=False),
    callback=lambda ctx, _, val: {hintstat.Statistic.COVERAGE: coverage.Coverage()}[val],
    multiple=True,
    required=False,
    help="Compute relevant statistics and store alongside .tsv",
)
def entrypoint(
    repo: list[pathlib.Path],
    output: pathlib.Path,
    statistic: list[hintstat.StatisticImpl] | None,
) -> None:
    merged_annotations = _collect(roots=repo)

    merged_annotations.write(output)

    for stat in statistic or []:
        statout = stat.forward(repos=repo, annotations=merged_annotations)


def _collect(
    roots: typing.Sequence[pathlib.Path],
) -> MergedAnnotations:

    assert len(set(roots)) == len(roots), "Cannot pass same folder twice!"
    assert len(set(map(lambda r: r.name, roots))) == len(
        roots
    ), "Your project roots must all be named differently!"

    results = list()

    # sanity check: these roots must be directories containing the same file tree
    files_per_root = dict()
    for root in roots:
        sroot = str(root)
        files_per_root[str(root)] = [
            os.path.relpath(subfile, sroot) for subfile in cstcli.gather_files([str(root)])
        ]

    roots_df = pd.concat(
        [pd.Series(sorted(files), name=root) for root, files in files_per_root.items()],
        axis=1,
    )
    deviating = roots_df[roots_df.apply(pd.Series.nunique, axis=1) != 1]

    assert deviating.empty, f"Differing folder structures, cannot compute hintdiff!"

    for root in roots:
        visitor = TypeCollectorVistor.strict(context=codemod.CodemodContext())
        result = codemod.parallel_exec_transform_with_prettyprint(
            transform=visitor,
            files=cstcli.gather_files([str(root)]),
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
        results.append(visitor.collection)

    paths_with_collections = [(r, col) for r, col in zip(roots, results)]
    merged_annotations = MergedAnnotations.from_collections(paths_with_collections)

    return merged_annotations


if __name__ == "__main__":
    entrypoint()
