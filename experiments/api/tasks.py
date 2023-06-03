import glob
import pathlib

import polars as pl

from scripts.common.schemas import TypeCollectionCategory


def tool_and_task(tool: str, task: TypeCollectionCategory) -> pathlib.Path:
    p = pathlib.Path(f"{tool}/{tool}@[{task}]+[{task}]")
    assert p.is_dir(), f"Could not find dataset for {p}"

    return p


def csv_graph(root: pathlib.Path) -> None:
    task_glob = f"{glob.escape(str(root))}/**/*.inferred.csv"
    pl.scan_csv(task_glob).show_graph()


def dump_polars(
    df: pl.DataFrame, tool: str, task: TypeCollectionCategory, author_proj: str
) -> None:
    p = pathlib.Path(tool) / str(task) / f"{author_proj}.csv"
    assert not p.is_file(), f"Preventing overwrite of already existing dataset: {p}"
    p.parent.mkdir(parents=True, exist_ok=True)

    df.write_csv(file=p)
