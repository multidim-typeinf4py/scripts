import operator
import pathlib

import pandas as pd
import pandera.typing as pt
from scripts.common.schemas import (
    ContextSymbolSchema,
    TypeCollectionCategory,
    TypeCollectionSchema,
    InferredSchema,
)


def context_vector_path(project: pathlib.Path) -> pathlib.Path:
    return project / "context-vectors.csv"


def write_context_vectors(
    df: pt.DataFrame[ContextSymbolSchema], project: pathlib.Path
) -> None:
    cpath = context_vector_path(project)
    cpath.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(cpath, index=False, header=list(ContextSymbolSchema.to_schema().columns))


def read_context_vectors(project: pathlib.Path) -> pt.DataFrame[ContextSymbolSchema]:
    cpath = context_vector_path(project)
    df = pd.read_csv(
        cpath,
        converters={
            "category": lambda c: operator.getitem(TypeCollectionCategory, c),
        },
    )

    return df.pipe(pt.DataFrame[ContextSymbolSchema])


def inferred_path(project: pathlib.Path) -> pathlib.Path:
    return project / "inferred.csv"


def write_inferred(df: pt.DataFrame[InferredSchema], project: pathlib.Path) -> None:
    ipath = inferred_path(project)
    df.to_csv(ipath, index=False)


def read_inferred(
    inpath: pathlib.Path, tool: str, task: list[TypeCollectionCategory]
) -> pt.DataFrame[InferredSchema]:
    outpath = inference_output_path(inpath, tool, task)
    ipath = inferred_path(outpath)
    df = pd.read_csv(
        ipath, converters={"category": lambda c: TypeCollectionCategory[c]}
    )

    return df.pipe(pt.DataFrame[InferredSchema])


def inference_output_path(
    outpath: pathlib.Path,
    tool: str,
    removed: list[TypeCollectionCategory],
) -> pathlib.Path:
    task_names = ",".join(map(str, removed))
    return outpath.parent / f"{tool}@[{task_names}]" / f"{outpath.name}"


def dataset_output_path(inpath: pathlib.Path, author_repo: str) -> pathlib.Path:
    assert inpath.is_dir(), f"Expected {inpath = } to be a folder to the dataset"
    return inpath / f"{author_repo}.csv"


def write_dataset(
    inpath: pathlib.Path, author_repo: str, df: pt.DataFrame[TypeCollectionSchema]
) -> None:
    opath = dataset_output_path(inpath, author_repo)
    print(f"Writing results to {opath}")
    opath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(opath, index=False)


def read_dataset(
    inpath: pathlib.Path, author_repo: str
) -> pt.DataFrame[TypeCollectionSchema]:
    rpath = dataset_output_path(inpath, author_repo)
    print(f"Reading from {rpath}")
    return pd.read_csv(
        rpath, converters={"category": lambda c: TypeCollectionCategory[c]}
    ).pipe(pt.DataFrame[TypeCollectionSchema])


def error_log_path(outpath: pathlib.Path) -> pathlib.Path:
    return outpath / "log.err"


def debug_log_path(outpath: pathlib.Path) -> pathlib.Path:
    return outpath / "log.dbg"


def info_log_path(outpath: pathlib.Path) -> pathlib.Path:
    return outpath / "log.inf"
