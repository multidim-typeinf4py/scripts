from tqdm import tqdm
import pandas as pd

from scripts.common.output import InferredIO, ExtendedDatasetIO, ContextIO
from scripts.common.schemas import (
    ContextCategory,
    ContextSymbolSchema,
    ExtendedTypeCollectionSchema,
    InferredSchema,
    TypeCollectionCategory,
    TypeCollectionSchema,
    SymbolSchema,
    RepositoryTypeCollectionSchema,
    RepositoryInferredSchema,
)

from scripts.dataset.normalisation import to_adjusted, to_base

from scripts.infer.structure import DatasetFolderStructure

import pandera.typing as pt
import pathlib


def load_entire_inferred(
    artifact_root: pathlib.Path,
    dataset: DatasetFolderStructure,
    tool_name: str,
    task: TypeCollectionCategory | str,
) -> pt.DataFrame[RepositoryInferredSchema]:
    dfs: list[pt.DataFrame[RepositoryInferredSchema]] = [
        RepositoryInferredSchema.example(size=0)
    ]

    for repository in (bar := tqdm(dataset.test_set())):
        inferred = InferredIO(
            artifact_root=artifact_root,
            dataset=dataset,
            repository=repository,
            tool_name=tool_name,
            task=task,
        )
        bar.set_description(str(inferred.full_location()))

        if inferred.full_location().exists():
            dfs.append(
                inferred.read().assign(repository=dataset.author_repo(repository))
            )

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
    batched[RepositoryInferredSchema.anno] = batched[
        RepositoryInferredSchema.anno
    ].replace("...", pd.NA)
    return batched


def load_groundtruths(
    artifact_root: pathlib.Path,
    dataset: DatasetFolderStructure,
) -> pt.DataFrame[RepositoryTypeCollectionSchema]:
    dfs: list[pt.DataFrame[RepositoryTypeCollectionSchema]] = [
        RepositoryTypeCollectionSchema.example(size=0)
    ]

    for repository in (bar := tqdm(dataset.test_set())):
        ground_truth = ExtendedDatasetIO(
            artifact_root=artifact_root,
            dataset=dataset,
            repository=repository,
        )
        bar.set_description(str(ground_truth.full_location()))
        if ground_truth.full_location().exists():
            assigned = (
                ground_truth.read()
                .assign(repository=str(dataset.author_repo(repository)))
                .pipe(pt.DataFrame[RepositoryTypeCollectionSchema])
            )
            dfs.append(assigned)

    # Remove duplicates
    assert dfs
    batched = pd.concat(dfs).drop_duplicates(
        subset=[
            RepositoryTypeCollectionSchema.repository,
            RepositoryTypeCollectionSchema.file,
            RepositoryTypeCollectionSchema.qname_ssa,
            RepositoryTypeCollectionSchema.category,
        ],
        keep=False,
    )

    # return pt.DataFrame[RepositoryTypeCollectionSchema](batched)

    # Remove symbols that are not annotatable directly
    contexts = list[pt.DataFrame[ContextSymbolSchema]]()
    for repository in (bar := tqdm(dataset.test_set())):
        context = ContextIO(
            artifact_root=artifact_root,
            dataset=dataset,
            repository=repository,
        )
        bar.set_description(str(context.full_location()))
        if context.full_location().exists():
            df = context.read().assign(repository=str(dataset.author_repo(repository)))
            contexts.append(df)

    assert contexts
    contexts_batch = pd.concat(contexts).drop_duplicates(
        subset=[
            RepositoryTypeCollectionSchema.repository,
            RepositoryTypeCollectionSchema.file,
            RepositoryTypeCollectionSchema.qname_ssa,
            RepositoryTypeCollectionSchema.category,
        ],
        keep=False,
    )
    #  print("context dfs:", contexts_batch.columns, contexts_batch.shape)

    gt_ctxt = pd.merge(
        left=batched,
        right=contexts_batch,
        how="inner",
        on=[
            "repository",
            SymbolSchema.file,
            SymbolSchema.category,
            SymbolSchema.qname,
            SymbolSchema.qname_ssa,
        ],
    )
    # print(gt_ctxt.shape)
    # Only look at directly annotatable
    gt_ctxt = gt_ctxt[
        gt_ctxt[ContextSymbolSchema.context_category].isin(
            [
                ContextCategory.ANN_ASSIGN,
                ContextCategory.SINGLE_TARGET_ASSIGN,
                ContextCategory.INSTANCE_ATTRIBUTE,
                ContextCategory.CALLABLE_RETURN,
                ContextCategory.CALLABLE_PARAMETER,
            ]
        )
    ]

    return pt.DataFrame[RepositoryTypeCollectionSchema](
        gt_ctxt.drop(
            columns=[
                ContextSymbolSchema.context_category,
                ContextSymbolSchema.loop,
                ContextSymbolSchema.reassigned,
                ContextSymbolSchema.nested,
                ContextSymbolSchema.flow_control,
                ContextSymbolSchema.import_source,
                ContextSymbolSchema.builtin_source,
                ContextSymbolSchema.local_source,
            ]
        )
    )


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
    comparable_anno: str,
    prediction_annos: str | list[str] = "anno",
) -> pd.DataFrame:
    ignored = list(
        {
            ExtendedTypeCollectionSchema.raw_anno,
            ExtendedTypeCollectionSchema.depth_limited_anno,
            ExtendedTypeCollectionSchema.base_anno,
            ExtendedTypeCollectionSchema.adjusted_anno,
        }.difference([comparable_anno])
    )
    select_anno = (
        truth.assign(trait_gt_form=truth[ExtendedTypeCollectionSchema.base_anno])
        .drop(columns=ignored)
        .rename(columns={comparable_anno: "gt_anno"})
    )

    # print(predictions.columns)
    # print(select_anno.columns)

    df = pd.merge(
        left=select_anno,
        right=predictions,
        how="left",
        on=[
            RepositoryInferredSchema.repository,
            TypeCollectionSchema.file,
            TypeCollectionSchema.category,
            TypeCollectionSchema.qname,
            TypeCollectionSchema.qname_ssa,
        ],
        # validate="1:1",
    )

    # Unify N/A representations
    if not isinstance(prediction_annos, list):
        prediction_annos = [prediction_annos, "gt_anno", "trait_gt_form"]
    else:
        prediction_annos = [*prediction_annos, "gt_anno", "trait_gt_form"]
    df[prediction_annos] = df[prediction_annos].fillna(pd.NA)
    return df


def evaluatable(
    joined: pd.DataFrame, clean_annos: str | list[str] = "anno"
) -> pd.DataFrame:
    # remove entries without truth
    missing_gt = joined["gt_anno"].isna()

    # do not track attributes, self, cls
    trivial_symbols = joined["qname"].str.endswith(
        (".self", ".cls", ".args", ".kwargs")
    )

    combined = ~missing_gt & ~trivial_symbols
    cleaned = joined.loc[combined]

    # change N/A to <MISSING> for evaluations
    cleaned[clean_annos] = cleaned[clean_annos].fillna("<MISSING>")
    return cleaned


def typet5_adjusted_form(
    df: pt.DataFrame[InferredSchema], anno: str = InferredSchema.anno
) -> pt.DataFrame[InferredSchema]:
    import tqdm

    tqdm.tqdm.pandas()

    # Replace mask artifacts
    df = df.copy()
    df[anno] = df[anno].replace(to_replace="...", value=pd.NA)

    # Dequalified
    df[anno] = df[anno].progress_apply(to_adjusted)

    # remove None, Any
    trivial_mask = df[anno].isin(["None", "Any"])
    df = df[~trivial_mask]

    return df


def typet5_base_form(
    df: pt.DataFrame[InferredSchema], anno: str = InferredSchema.anno
) -> pt.DataFrame[InferredSchema]:
    import tqdm

    tqdm.tqdm.pandas()

    # Replace mask artifacts
    df = df.copy()
    df[anno] = df[anno].replace(to_replace="...", value=pd.NA)

    # Dequalified
    df[anno] = df[anno].progress_apply(to_base)

    # remove None, Any
    trivial_mask = df[anno].isin(["None", "Any"])
    df = df[~trivial_mask]

    return df
