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
)
@click.option(
    "-o",
    "--outpath",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path),
    required=True,
    help="Folder for annotation dataframe to be written into",
)
def cli_entrypoint(dataset: pathlib.Path, outpath: pathlib.Path) -> None:
    structure = DatasetFolderStructure(dataset_root=dataset)
    print(dataset, structure)

    test_set = structure.test_set()
    for project, subset in (pbar := tqdm.tqdm(test_set.items())):
        pbar.set_description(desc=f"Collecting from {project}")
        collection = build_type_collection(root=project, allow_stubs=False, subset=subset).df

        ar = structure.author_repo(project)
        author_repo = f"{ar['author']}.{ar['repo']}"

        output.write_dataset(outpath, author_repo, df=collection)


if __name__ == "__main__":
    cli_entrypoint()
