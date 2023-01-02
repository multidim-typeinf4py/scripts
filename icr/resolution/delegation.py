import functools
import pathlib

from ._base import BatchResolution
from common.schemas import InferredSchema, SymbolSchema

import pandas as pd
import pandera.typing as pt

import enum


class DelegationOrder(enum.IntEnum):
    STATIC = enum.auto()
    DYNAMIC = enum.auto()
    PROBABILISTIC = enum.auto()


class Delegation(BatchResolution):
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
            match o:
                case DelegationOrder.STATIC:
                    ordered.append(static)
                case DelegationOrder.DYNAMIC:
                    ordered.append(dynamic)
                case DelegationOrder.PROBABILISTIC:
                    ordered.append(probabilistic)

        ordered = list(filter(len, ordered))

        # Remove all predictions where no prediction was made,
        # Then retain the first occurrence of every symbol with a hint in a given file
        covered = (
            pd.concat(ordered)
            .dropna(subset="anno")
            .drop_duplicates(subset=list(self.reference.columns), keep="first")
        ).pipe(pt.DataFrame[InferredSchema])

        return covered