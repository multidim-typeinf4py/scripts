import pathlib

import click


from common import TypeCollection
from .collector import build_type_collection


@click.command(
    name="symbols",
    short_help="Traverse the given repository and collect annotated tokens",
)
@click.option(
    "-r",
    "--root",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    required=True,
    help="Root of Repository to gather annotations from",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    required=True,
    help="Output path for .tsv",
)
def entrypoint(root: pathlib.Path, output: pathlib.Path) -> None:
    collection = build_type_collection(root)
    _store(collection, output)


def _store(collection: TypeCollection, output: pathlib.Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    collection.write(output)


if __name__ == "__main__":
    entrypoint()
