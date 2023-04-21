import pathlib
import click

from . import DatasetConsumer
from .manytypes4py import ManyTypes4PyConsumerFull, ManyTypes4PyConsumerNeat


from common import output


@click.command(name="dataset", help="Consume dataset into inference agnostic DataFrame")
@click.option(
    "-i",
    "--inpath",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path),
)
@click.option(
    "-k", "--kind", type=click.Choice(["manytypes4py", "manytypes4py-full"], case_sensitive=False)
)
def cli_entrypoint(inpath: pathlib.Path, kind: str) -> None:
    consumer: DatasetConsumer
    if kind == "manytypes4py":
        consumer = ManyTypes4PyConsumerNeat(
            duplicates=inpath / "duplicate_files.txt",
            type_checked_files=inpath / "type_checked_files.txt",
            split=inpath / "dataset_split.csv",
        )
        dataset = inpath / "repos"
    elif kind == "manytypes4py-full":
        consumer = ManyTypes4PyConsumerFull(split=inpath / "dataset_split.csv")
        dataset = inpath / "repos"
    else:
        assert f"Unknown kind: {kind}"

    dataset = consumer.produce(dataset=dataset)
    output.write_dataset(inpath, kind, dataset)


if __name__ == "__main__":
    cli_entrypoint()
