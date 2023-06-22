import functools

import click
import pathlib
import tqdm

import libcst
from libcst import codemod, matchers as m

from scripts.common import output, extending
from scripts.common.schemas import (
    InferredSchema,
    TypeCollectionSchema,
    ExtendedTypeCollectionSchema,
)

from scripts.infer.inference import PyreQuery
from scripts.infer.structure import DatasetFolderStructure

from scripts.infer.annotators.normalisation import Normalisation, Normaliser

from scripts.symbols.collector import build_type_collection
from scripts import utils

import pandas as pd
from pandera import typing as pt


@click.command(name="dataset", help="Consume dataset into inference agnostic DataFrame")
@click.option(
    "-d",
    "--dataset",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path),
    required=True,
)
@click.option(
    "-o",
    "--outpath",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path),
    required=True,
    help="Folder for type collection dataframe to be written into",
)
@click.option(
    "-r",
    "--overwrite",
    required=False,
    default=False,
    help="Overwrite existing results in output folder",
    is_flag=True,
)
def cli_entrypoint(
    dataset: pathlib.Path, outpath: pathlib.Path, overwrite: bool
) -> None:
    structure = DatasetFolderStructure(dataset_root=dataset)
    print(structure)

    normalisation_strategy = Normalisation.default()

    test_set = structure.test_set()
    for project, subset in (pbar := tqdm.tqdm(test_set.items())):
        pbar.set_description(desc=f"Collecting from {project}")

        dataset_io = output.DatasetIO(artifact_root=outpath, dataset=structure, repository=project)
        if not overwrite and dataset_io.full_location().exists():
            print(
                f"Skipping {project}; dataset already exists, no extension requested, and overwrite flag was not provided!"
            )
            continue

        else:
            with utils.scratchpad(project) as sc:
                # Normalise codebase annotations
                res = codemod.parallel_exec_transform_with_prettyprint(
                    transform=Normaliser(codemod.CodemodContext(), normalisation_strategy),
                    jobs=utils.worker_count(),
                    repo_root=sc,
                    files=[str(sc / f) for f in subset],
                )
                print(utils.format_parallel_exec_result("Normalising codebase", res))

            collection = build_type_collection(
                root=project,
                allow_stubs=False,
                subset=subset,
            ).df
            dataset_io.write(collection)


if __name__ == "__main__":
    tqdm.pandas()
    cli_entrypoint()
