import pathlib

import click
import tqdm
from libcst import codemod

from scripts import utils
from scripts.common.output import ContextIO
from scripts.common.schemas import (
    ExtendedTypeCollectionSchema,
)
from scripts.infer.annotators.normalisation import Normaliser, Normalisation


@click.command(
    name="context", help="Create vectors to classify contexts of annotatables"
)
@click.option(
    "-d",
    "--dataset",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    required=True,
    help="Dataset to iterate over (can also be a singular project!)",
)
@click.option(
    "-o",
    "--outpath",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    required=True,
    help="Base folder for inference results to be written into",
)
@click.option(
    "-w",
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite an existing output folder from previous run",
)
def cli_entrypoint(
    dataset: pathlib.Path,
    outpath: pathlib.Path,
    overwrite: bool,
) -> None:
    from scripts.context import RelevantFeatures
    from scripts.context.visitors import generate_context_vectors
    from scripts.infer.structure import DatasetFolderStructure

    structure = DatasetFolderStructure(dataset)
    test_set = structure.test_set()

    for project, subset in (pbar := tqdm.tqdm(test_set.items())):
        pbar.set_description(desc=f"{project}")

        output_io = ContextIO(
            artifact_root=outpath,
            dataset=structure,
            repository=project,
        )

        if not overwrite and output_io.full_location().exists():
            print(
                f"Skipping computing context on '{project}'; "
                f"artifact exists and --overwrite was not specified"
            )
            continue

        # Safety first!
        with utils.scratchpad(project) as sc:
            context_vectors = generate_context_vectors(
                features=RelevantFeatures.default(),
                project=sc,
                subset=subset,
            )

        print(f"Writing context vectors to {output_io.full_location()}")
        output_io.write(context_vectors)


if __name__ == "__main__":
    cli_entrypoint()
