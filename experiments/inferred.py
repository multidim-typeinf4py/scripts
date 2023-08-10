import pandas as pd

from scripts.common.output import InferredIO, ExtendedDatasetIO
from scripts.common.schemas import (
    ExtendedTypeCollectionSchema,
    InferredSchema,
    TypeCollectionCategory,
    TypeCollectionSchema,
    SymbolSchema,
    RepositoryTypeCollectionSchema,
    RepositoryInferredSchema,
)

from scripts.dataset.normalisation import to_adjusted

from scripts.infer.structure import DatasetFolderStructure

import pandera.typing as pt
import pathlib


def load_entire_inferred(
    artifact_root: pathlib.Path,
    dataset: DatasetFolderStructure,
    tool_name: str,
    task: TypeCollectionCategory | str,
) -> pt.DataFrame[RepositoryInferredSchema]:
    dfs: list[pt.DataFrame[InferredSchema]] = [InferredSchema.example(size=0)]

    for repository in dataset.test_set():
        inferred = InferredIO(
            artifact_root=artifact_root,
            dataset=dataset,
            repository=repository,
            tool_name=tool_name,
            task=task,
        )
        if inferred.full_location().exists():
            dfs.append(inferred.read().assign(repository=dataset.author_repo(repository)))

    batched = (
        pd.concat(dfs)
        .drop_duplicates(
            subset=[
                ExtendedTypeCollectionSchema.file,
                ExtendedTypeCollectionSchema.qname_ssa,
                ExtendedTypeCollectionSchema.category,
            ],
            keep=False,
        )
        .pipe(pt.DataFrame[RepositoryInferredSchema])
    )
    return batched


def load_groundtruths(
    artifact_root: pathlib.Path,
    dataset: DatasetFolderStructure,
) -> pt.DataFrame[RepositoryTypeCollectionSchema]:
    dfs: list[pt.DataFrame[RepositoryTypeCollectionSchema]] = [
        RepositoryTypeCollectionSchema.example(size=0)
    ]

    for repository in dataset.test_set():
        ground_truth = ExtendedDatasetIO(
            artifact_root=artifact_root,
            dataset=dataset,
            repository=repository,
        )
        if ground_truth.full_location().exists():
            assigned = (
                ground_truth.read()
                .assign(repository=dataset.author_repo(repository))
                .pipe(pt.DataFrame[RepositoryTypeCollectionSchema])
            )
            dfs.append(assigned)

    # Remove duplicates
    batched = pd.concat(dfs).drop_duplicates(
        subset=[
            RepositoryTypeCollectionSchema.repository,
            RepositoryTypeCollectionSchema.file,
            RepositoryTypeCollectionSchema.qname_ssa,
            RepositoryTypeCollectionSchema.category,
        ],
        keep=False,
    )
    return batched.pipe(pt.DataFrame[RepositoryTypeCollectionSchema])


def error_if_duplicate_keys(df: pt.DataFrame[SymbolSchema]) -> None:
    keys = [
        TypeCollectionSchema.file,
        TypeCollectionSchema.category,
        TypeCollectionSchema.qname,
        TypeCollectionSchema.qname_ssa,
    ]
    if RepositoryTypeCollectionSchema.repository in df.columns:
        keys.append(RepositoryTypeCollectionSchema.repository)

    duplicate_keys = df.duplicated(
        subset=keys,
        keep=False,
    )
    if duplicate_keys.any():
        raise RuntimeError(f"Duplicate keys in truth set:\n{df[duplicate_keys]}")


def join_truth_to_preds(
    truth: pt.DataFrame[RepositoryTypeCollectionSchema],
    predictions: pt.DataFrame[RepositoryInferredSchema],
) -> pd.DataFrame:
    select_anno = truth.drop(
        columns=[
            ExtendedTypeCollectionSchema.raw_anno,
            ExtendedTypeCollectionSchema.depth_limited_anno,
            ExtendedTypeCollectionSchema.base_anno,
        ]
    ).rename({ExtendedTypeCollectionSchema.adjusted_anno: "anno"})

    df = pd.merge(
        left=select_anno,
        right=predictions,
        on=[
            RepositoryInferredSchema.repository,
            TypeCollectionSchema.file,
            TypeCollectionSchema.category,
            TypeCollectionSchema.qname,
            TypeCollectionSchema.qname_ssa,
        ],
        # validate="1:1",
    )
    return df


def typet5_adjusted_form(
    df: pt.DataFrame[InferredSchema], anno: str = InferredSchema.anno
) -> pt.DataFrame[InferredSchema]:
    import tqdm

    tqdm.tqdm.pandas()

    # Replace mask artifacts
    df[anno] = df[anno].replace(to_replace="...", value=pd.NA)

    # Dequalified
    df[anno] = df[anno].progress_apply(to_adjusted)

    # remove None, Any
    trivial_mask = df[anno].isin(["None", "Any"])
    df = df.drop(df[trivial_mask].index)

    return df