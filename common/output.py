import operator
import pathlib

import pandas as pd
import pandera.typing as pt
from common.schemas import (
    ContextSymbolSchema,
    TypeCollectionCategory,
    InferredSchema,
)

from infer.inference import Inference


def context_vector_path(project: pathlib.Path) -> pathlib.Path:
    return project / ".context-vectors.csv"


def write_context_vectors(df: pt.DataFrame[ContextSymbolSchema], project: pathlib.Path) -> None:
    cpath = context_vector_path(project)
    cpath.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(cpath, index=False, header=list(ContextSymbolSchema.to_schema().columns))


def read_context_vectors(project: pathlib.Path) -> pt.DataFrame[ContextSymbolSchema]:
    cpath = context_vector_path(project)
    df = pd.read_csv(
        cpath,
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
        index=False,
        header=InferredSchema.to_schema().columns,
    )


def read_icr(project: pathlib.Path) -> pt.DataFrame[InferredSchema]:
    ipath = icr_path(project)
    df = pd.read_csv(ipath, converters={"category": lambda c: TypeCollectionCategory[c]})

    return df.pipe(pt.DataFrame[InferredSchema])


def inference_output_path(
    inpath: pathlib.Path,
    tool: str,
) -> pathlib.Path:
    return inpath.parent / f"{inpath.name}@({tool})"
