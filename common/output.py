import operator
import pathlib

import pandas as pd
import pandera.typing as pt
from common.schemas import (
    ContextCategory,
    ContextSymbolSchema,
    TypeCollectionCategory,
    InferredSchema,
    InferredSchemaColumns,
)


def context_path(project: pathlib.Path) -> pathlib.Path:
    return project / ".context.csv"


def write_context(df: pt.DataFrame[ContextSymbolSchema], project: pathlib.Path) -> None:
    cpath = context_path(project)
    cpath.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(cpath, sep="\t", index=False, header=ContextSymbolSchema.to_schema().columns)


def read_context(project: pathlib.Path) -> pt.DataFrame[ContextSymbolSchema]:
    cpath = context_path(project)
    df = pd.read_csv(
        cpath,
        sep="\t",
        converters={
            "category": lambda c: operator.getitem(TypeCollectionCategory, c),
            # "ctxt_category": lambda c: operator.getitem(ContextCategory, c),
        },
    )

    return df.pipe(pt.DataFrame[ContextSymbolSchema])


def icr_path(project: pathlib.Path) -> pathlib.Path:
    return project / ".icr.csv"


def write_icr(df: pt.DataFrame[InferredSchema], project: pathlib.Path) -> None:
    ipath = icr_path(project)
    df.to_csv(
        ipath,
        sep="\t",
        index=False,
        header=InferredSchema.to_schema().columns,
    )


def read_icr(project: pathlib.Path) -> pt.DataFrame[InferredSchema]:
    ipath = icr_path(project)
    df = pd.read_csv(ipath, sep="\t", converters={"category": lambda c: TypeCollectionCategory[c]})

    return df.pipe(pt.DataFrame[InferredSchema])
