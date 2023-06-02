import click
import pathlib
import tqdm

from src.symbols.collector import build_type_collection

from src.infer.inference._base import DatasetFolderStructure

from src.common import output


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
    structure = DatasetFolderStructure.from_folderpath(dataset)
    print(dataset, structure)

    test_set = {p: s for p, s in structure.test_set(dataset).items() if p.is_dir()}
    for project, subset in (pbar := tqdm.tqdm(test_set.items())):
        pbar.set_description(desc=f"Collecting from {project}")
        collection = build_type_collection(root=project, allow_stubs=False, subset=subset).df

        ar = structure.author_repo(project)
        author_repo = f"{ar['author']}.{ar['repo']}"

        output.write_dataset(outpath, author_repo, df=collection)


if __name__ == "__main__":
    cli_entrypoint()
