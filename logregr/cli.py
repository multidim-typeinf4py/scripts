import pathlib

import pandas as pd
import click

from common.schemas import ContextSymbolSchema
from common import output

from symbols.collector import build_type_collection


@click.group(
    name="logregr",
    help="Interact with logistic regression model to evaluate efficacy of inference tools",
)
def entrypoint():
    ...


@entrypoint.command(name="train", help="Train logistic regression model")
@click.option(
    "-i",
    "--inpath",
    type=click.Tuple(
        (
            click.Path(exists=True, dir_okay=True, path_type=pathlib.Path),
            click.Path(exists=True, dir_okay=True, path_type=pathlib.Path),
            str,
        )
    ),
    help="Projects to use as data; tuples of (ground-truth, tool-annotated, method)",
    multiple=True,
)
def train(inpath: list[tuple[pathlib.Path, pathlib.Path]]) -> None:
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
        how="left",
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
    print(features_on_symbols[[ContextSymbolSchema.qname_ssa, "anno_gt", "anno_ta", "score"]])

    features_on_symbols.loc[matching_annos & ~missing_annos, "score"] = 1

    # Do not reward not matching ground truth
    features_on_symbols.loc[~matching_annos & ~missing_annos, "score"] = 0
    
    # print(features_on_symbols)

    features_on_symbols = ta_df[
        [
            ContextSymbolSchema.loop,
            ContextSymbolSchema.reassigned,
            ContextSymbolSchema.nested,
            ContextSymbolSchema.user_defined,
            ContextSymbolSchema.ctxt_category,
        ]
    ]
    print(features_on_symbols.dtypes)
