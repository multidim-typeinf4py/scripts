import pathlib

import click

from .analyses import symbol_commonality, annotation_commonality
from symbols.collector import build_type_collection


@click.command(name="harness")
@click.option(
    "-b",
    "--baseline",
    help="Path to ground truth",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path),
)
@click.option(
    "-i",
    "--inferred",
    help="Path to codebase created by inference tool",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path),
)
def cli_entrypoint(baseline: pathlib.Path, inferred: pathlib.Path) -> None:
    baseline_collection = build_type_collection(baseline).df
    inferred_collection = build_type_collection(inferred).df

    sym = symbol_commonality(baseline_collection, inferred_collection)
    print(f"{len(sym.common)} common symbols")
    print(f"{len(sym.only_in_baseline)} only in baseline:\n{sym.only_in_baseline}")
    print(f"{len(sym.only_in_inferred)} only in inferred:\n{sym.only_in_inferred}")

    anno = annotation_commonality(baseline_collection, inferred_collection)
    print(f"{len(anno.exact_match)} exact match")
    print(f"{len(anno.missing)} missing:\n{anno.missing}")
    print(f"{len(anno.parametric_match)} parametric match:\n{anno.parametric_match}")
    print(f"{len(anno.differing)} differ entirely:\n{anno.differing}")


if __name__ == "__main__":
    cli_entrypoint()
