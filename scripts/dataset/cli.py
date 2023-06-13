import click
import pathlib
import tqdm

from scripts.symbols.collector import build_type_collection

from scripts.infer.structure import DatasetFolderStructure

from scripts.common import output


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
)
def cli_entrypoint(dataset: pathlib.Path, outpath: pathlib.Path, overwrite: bool) -> None:
    structure = DatasetFolderStructure(dataset_root=dataset)
    print(structure)

    test_set = structure.test_set()
    for project, subset in (pbar := tqdm.tqdm(test_set.items())):
        dataset_io = output.DatasetIO(
            artifact_root=outpath,
            dataset=structure,
            repository=project
        )

        if not overwrite and dataset_io.full_location().exists():
            print(f"Skipping {project}; dataset already exists and overwrite flag was not provided!")
            continue

        pbar.set_description(desc=f"Collecting from {project}")
        collection = build_type_collection(root=project, allow_stubs=False, subset=subset).df

        dataset_io.write(collection)

if __name__ == "__main__":
    cli_entrypoint()
