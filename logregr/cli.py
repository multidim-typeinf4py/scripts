import pathlib
import typing

import pandas as pd
import pandera.typing as pt

import click

from common.schemas import ContextSymbolSchema, ContextDatasetSchema, TypeCollectionCategory
from common import output

from symbols.collector import build_type_collection


from infer.inference import HiTyper, PyreInfer, PyreQuery, Type4PyN1, TypeWriter


@click.group(
    name="logregr",
    help="Interact with logistic regression model to evaluate efficacy of inference tools",
)
def cli_entrypoint():
    ...


SUPPORTED = dict(
    (method, method_id)
    for method_id, method in enumerate(
        [HiTyper.method, PyreInfer.method, PyreQuery.method, Type4PyN1.method, TypeWriter.method]
    )
)


@cli_entrypoint.command(name="train", help="Train logistic regression model")
@click.option(
    "-i",
    "--inpath",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path),
)
def train(inpath: pathlib.Path) -> None:
    ...


@cli_entrypoint.command(name="dataset", help="Create dataset for model")
@click.option(
    "-i",
    "--inpath",
    type=click.Tuple(
        (
            click.Path(exists=True, dir_okay=True, path_type=pathlib.Path),
            click.Path(exists=True, dir_okay=True, path_type=pathlib.Path),
            click.Choice(choices=list(SUPPORTED.keys())),
        )
    ),
    help="Projects to use as data; tuples of (ground-truth, tool-annotated, method)",
    multiple=True,
)
@click.option(
    "-a",
    "--append-to",
    type=click.Path(path_type=pathlib.Path),
    help="Append newly context vector dataset to existing dataset",
    required=False,
)
def dataset(
    inpath: list[tuple[pathlib.Path, pathlib.Path, str]], append_to: typing.Union[pathlib.Path, None]
) -> None:
    ground_truth_df = [build_type_collection(ip).df for ip, _, _ in inpath]

    # Remove all NAs from the ground truth, if there are any, as we cannot use them for training
    gt_df = pd.concat(ground_truth_df).dropna()

    tool_annotated_dfs = [
        output.read_context_vectors(ip).assign(method=method) for _, ip, method in inpath
    ]
    ta_df = pd.concat(tool_annotated_dfs)

    features_on_symbols = pd.merge(
        left=gt_df,
        right=ta_df,
        how="right",
        on=[
            ContextSymbolSchema.file,
            ContextSymbolSchema.category,
            ContextSymbolSchema.qname,
            ContextSymbolSchema.qname_ssa,
        ],
        suffixes=("_gt", "_ta"),
    )

    # Penalise lacking predictions
    missing_annos = features_on_symbols["anno_ta"].isna()

    # Reward matching ground truth
    matching_annos = (
        features_on_symbols.loc[~missing_annos, "anno_gt"]
        == features_on_symbols.loc[~missing_annos, "anno_ta"]
    )
    features_on_symbols.loc[missing_annos, "score"] = -1
    features_on_symbols.loc[matching_annos & ~missing_annos, "score"] = 1

    # Do not reward not matching ground truth
    features_on_symbols.loc[~matching_annos & ~missing_annos, "score"] = 0
    features_on_symbols["score"] = features_on_symbols["score"].astype(int)

    dataset = features_on_symbols[
        [
            "method",
            ContextSymbolSchema.file,
            ContextSymbolSchema.qname_ssa,
            "anno_gt",
            "anno_ta",
            "score",
            ContextSymbolSchema.loop,
            ContextSymbolSchema.reassigned,
            ContextSymbolSchema.nested,
            ContextSymbolSchema.builtin,
            ContextSymbolSchema.ctxt_category,
        ]
    ].pipe(pt.DataFrame[ContextDatasetSchema])

    if append_to is None:
        print("--append-to was not given; exiting...")
        return

    if not append_to.is_file():
        print(f"Creating dataset at {append_to}")
        append_to.parent.mkdir(parents=True, exist_ok=True)
        append_to.touch(exist_ok=True)
        df = ContextDatasetSchema.to_schema().example(size=0)

    else:
        df = pd.read_csv(
            append_to, converters={"category": lambda c: TypeCollectionCategory[c]}
        ).pipe(pt.DataFrame[ContextDatasetSchema])

    df = pd.concat([df, dataset], ignore_index=True)

    print(f"New dataset size: {df.shape}; writing to {append_to}")
    df.to_csv(append_to, index=False, header=ContextDatasetSchema.to_schema().columns)
