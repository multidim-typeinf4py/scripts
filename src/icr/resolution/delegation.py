import pathlib

from ._base import BatchResolution
from src.common import InferredSchema, SymbolSchema

import pandas as pd
import pandera.typing as pt

import enum


class DelegationOrder(enum.IntEnum):
    STATIC = enum.auto()
    DYNAMIC = enum.auto()
    PROBABILISTIC = enum.auto()


class Delegation(BatchResolution):
    method = "delegation"

    def __init__(
        self,
        project: pathlib.Path,
        reference: pt.DataFrame[SymbolSchema],
        order: tuple[DelegationOrder, DelegationOrder, DelegationOrder],
    ) -> None:
        super().__init__(project, reference)
        self.order = order

    def forward(
        self,
        static: pt.DataFrame[InferredSchema],
        dynamic: pt.DataFrame[InferredSchema],
        probabilistic: pt.DataFrame[InferredSchema],
    ) -> pt.DataFrame[InferredSchema]:
        ordered = list()

        for o in self.order:
            if o is DelegationOrder.STATIC:
                ordered.append(static)
            elif o is DelegationOrder.DYNAMIC:
                ordered.append(dynamic)
            elif o is DelegationOrder.PROBABILISTIC:
                ordered.append(probabilistic)

        ordered = list(filter(len, ordered))
        ordered_df = pd.concat(ordered, ignore_index=True)

        # Remove all predictions where no prediction was made,
        # Then retain the first occurrence of every symbol with a hint in a given file
        covered = ordered_df.dropna(subset="anno").drop_duplicates(
            subset=[
                InferredSchema.file,
                InferredSchema.category,
                InferredSchema.qname_ssa,
            ],
            keep="first",
        )

        # If symbol is missing after dropping all that, that means all agents did not make a prediction for the symbol
        uniq_ordered = ordered_df.drop_duplicates(
            subset=[
                InferredSchema.file,
                InferredSchema.category,
                InferredSchema.qname_ssa,
            ],
            keep="first",
        )
        missing = pd.concat((covered, uniq_ordered), ignore_index=True).drop_duplicates(
            subset=[
                InferredSchema.file,
                InferredSchema.category,
                InferredSchema.qname_ssa,
            ],
            keep=False,
        )
        missing_method_tag = "+".join(o[InferredSchema.method].iloc[0] for o in ordered)

        restored = pd.concat(
            (
                covered,
                missing.assign(
                    method=missing_method_tag, anno=BatchResolution.UNRESOLVED
                ),
            ),
            ignore_index=True,
        )
        return restored.pipe(pt.DataFrame[InferredSchema])
