import pathlib

import click
import tqdm
from libcst import codemod

from pandera import typing as pt

from scripts import utils
from scripts.common import output
from scripts.common.schemas import (
    TypeCollectionSchema,
    ExtendedTypeCollectionSchema,
)
from scripts.infer.annotators.normalisation import Normalisation, Normaliser
from scripts.infer.structure import DatasetFolderStructure
from scripts.symbols.collector import build_type_collection

from .normalisation import to_limited, to_adjusted, to_base


@click.command(name="dataset", help="Consume dataset into inference agnostic DataFrame")
@click.option(
    "-d",
    "--dataset",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    required=True,
)
@click.option(
    "-o",
    "--outpath",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
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
@click.option(
    "-e",
    "--extended",
    required=False,
    default=False,
    help="Compute extended dataset form additionally, with TypeT5's label processing applied",
    is_flag=True,
)
def cli_entrypoint(
    dataset: pathlib.Path,
    outpath: pathlib.Path,
    overwrite: bool,
    extended: bool,
) -> None:
    structure = DatasetFolderStructure(dataset_root=dataset)
    print(structure)

    normalisation_strategy = Normalisation.default()

    test_set = structure.test_set()
    for project, subset in (pbar := tqdm.tqdm(test_set.items())):
        pbar.set_description(desc=f"Collecting from {project}")

        dataset_io = output.DatasetIO(
            artifact_root=outpath, dataset=structure, repository=project
        )
        if not overwrite and not extended and dataset_io.full_location().exists():
            print(
                f"Skipping {project}; dataset already exists, no extension requested, and overwrite flag was not provided!"
            )
            continue

        elif extended and dataset_io.full_location().exists():
            print("Loading ground truth from cache for extended")
            collection = dataset_io.read()

        else:
            with utils.scratchpad(project) as sc:
                # Normalise codebase annotations
                res = codemod.parallel_exec_transform_with_prettyprint(
                    transform=Normaliser(
                        codemod.CodemodContext(), normalisation_strategy
                    ),
                    jobs=utils.worker_count(),
                    repo_root=sc,
                    files=[str(sc / f) for f in subset],
                )
                print(utils.format_parallel_exec_result("Normalising codebase", res))

                collection = build_type_collection(
                    root=sc,
                    allow_stubs=False,
                    subset=subset,
                ).df
            dataset_io.write(collection)

        extended_dataset_io = output.ExtendedDatasetIO(
            artifact_root=outpath, dataset=structure, repository=project
        )
        if not extended:
            print("Not computing extended form; request with --extended")
            continue

        elif not overwrite and extended_dataset_io.full_location().exists():
            print(
                "Skipping computing extended form; "
                "already exists and --overwrite was not specified"
            )

        extended_df = collection.rename(
            columns={TypeCollectionSchema.anno: ExtendedTypeCollectionSchema.raw_anno}
        )
        extended_df[ExtendedTypeCollectionSchema.depth_limited_anno] = extended_df[
            ExtendedTypeCollectionSchema.raw_anno
        ].progress_apply(lambda a: to_limited(a))

        extended_df[ExtendedTypeCollectionSchema.adjusted_anno] = extended_df[
            ExtendedTypeCollectionSchema.depth_limited_anno
        ].progress_apply(lambda a: to_adjusted(a))

        extended_df[ExtendedTypeCollectionSchema.base_anno] = extended_df[
            ExtendedTypeCollectionSchema.depth_limited_anno
        ].progress_apply(lambda a: to_base(a))

        extended_dataset_io.write(extended_df.pipe(
            pt.DataFrame[ExtendedTypeCollectionSchema]
        ))


if __name__ == "__main__":
    tqdm.pandas()
    cli_entrypoint()
