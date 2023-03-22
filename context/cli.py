import enum
import pathlib

import click
import pandera.typing as pt
import pandas as pd
import libcst.codemod as codemod

from common.schemas import ContextSymbolSchema
from common import output

from context.features import RelevantFeatures

from .visitors import generate_context_vectors_for_project


class Purpose(str, enum.Enum):
    LIBRARY = "library"
    APPLICATION = "appl"


@click.command(name="context", help="Create vectors to classify contexts of annotatables")
@click.option(
    "-i",
    "--inpath",
    type=click.Path(exists=True, dir_okay=True, file_okay=True, path_type=pathlib.Path),
    help="Project to analyse",
)
@click.option(
    "-l",
    "--loop",
    is_flag=True,
    show_default=True,
    default=False,
    help="1 iff annotatable is in a loop else 0",
)
@click.option(
    "-r",
    "--reassigned",
    is_flag=True,
    show_default=True,
    default=False,
    help="1 iff variable is assigned to multiple times within same scope else 0",
)
@click.option(
    "-n",
    "--nested",
    is_flag=True,
    show_default=True,
    default=False,
    help="1 iff annotatable is in a nested scope else 0",
)
@click.option(
    "-f",
    "--flow",
    is_flag=True,
    show_default=True,
    default=False,
    help="1 iff annotatable is in a branch (if-else) else 9",
)
@click.option(
    "-u",
    "--user-defined",
    is_flag=True,
    show_default=True,
    default=False,
    help="1 iff given annotation is user-defined else 0",
)
def cli_entrypoint(
    inpath: pathlib.Path,
    loop: bool,
    reassigned: bool,
    nested: bool,
    flow: bool,
    user_defined: bool,
) -> None:
    features = RelevantFeatures(
        loop=loop,
        reassigned=reassigned,
        nested=nested,
        user_defined=user_defined,
        branching=flow,
    )

    df = generate_context_vectors_for_project(features, repo=inpath)
    print(f"Feature set size: {df.shape}; writing to {output.context_vector_path(inpath)}")
    output.write_context_vectors(df, inpath)


if __name__ == "__main__":
    cli_entrypoint()
