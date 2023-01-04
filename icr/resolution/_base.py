import abc
from dataclasses import dataclass
import pathlib

from common.schemas import (
    SymbolSchema,
    InferredSchema,
    InferredSchemaColumns,
    TypeCollectionCategory,
)

import pandas as pd
from pandas._libs import missing
import pandera.typing as pt


@dataclass(frozen=True)
class Metadata:
    file: str
    category: TypeCollectionCategory
    qname: str


class ConflictResolution(abc.ABC):
    UNRESOLVED = missing.NA

    @property
    @abc.abstractmethod
    def method() -> str:
        ...

    def __init__(self, project: pathlib.Path, reference: pt.DataFrame[SymbolSchema]) -> None:
        super().__init__()
        self.project = project
        self.reference = reference

    def resolve(
        self,
        static: pt.DataFrame[InferredSchema] | None = None,
        dynamic: pt.DataFrame[InferredSchema] | None = None,
        probabilistic: pt.DataFrame[InferredSchema] | None = None,
    ) -> pt.DataFrame[InferredSchema]:
        # Defaulting
        static = (
            static
            if static is not None
            else pt.DataFrame[InferredSchema](columns=InferredSchemaColumns)
        )
        dynamic = (
            dynamic
            if dynamic is not None
            else pt.DataFrame[InferredSchema](columns=InferredSchemaColumns)
        )
        probabilistic = (
            probabilistic
            if probabilistic is not None
            else pt.DataFrame[InferredSchema](columns=InferredSchemaColumns)
        )

        # Discover common symbols
        ## TODO: The below is caused by using stub files, consider dismissal of the below

        # TODO: This does not quite work, as class attributes are added to classes
        # TODO: that are likely not present in the source file

        # TODO: So how do we determine a fitting baseline?
        how = "right"
        common_cols = ["file", "qname", "category"]

        static = pd.merge(left=static, right=self.reference, how=how, on=common_cols)
        dynamic = pd.merge(left=dynamic, right=self.reference, how=how, on=common_cols)
        probabilistic = pd.merge(left=probabilistic, right=self.reference, how=how, on=common_cols)

        inferred = self._resolve(static, dynamic, probabilistic)

        # Readd symbols with unresolved that were removed due to no
        # tool making a prediction
        method_names = [inf["method"].iloc[0] for inf in [static, dynamic, probabilistic] if len(inf)]
        readd = self.reference.assign(
            method="+".join(method_names), anno=BatchResolution.UNRESOLVED
        )

        return (
            pd.concat([inferred, readd], ignore_index=True)
            .drop_duplicates(subset=list(self.reference.columns), keep="first")
            .pipe(pt.DataFrame[InferredSchema])
        )


    @abc.abstractmethod
    def _resolve(
        self,
        static: pt.DataFrame[InferredSchema],
        dynamic: pt.DataFrame[InferredSchema],
        probabilistic: pt.DataFrame[InferredSchema],
    ) -> pt.DataFrame[InferredSchema]:
        ...


class BatchResolution(ConflictResolution):
    """Process all given DataFrames at once, forgoeing iteration"""

    def _resolve(
        self,
        static: pt.DataFrame[InferredSchema],
        dynamic: pt.DataFrame[InferredSchema],
        probabilistic: pt.DataFrame[InferredSchema],
    ) -> pt.DataFrame[InferredSchema]:
        return self.forward(static, dynamic, probabilistic)

    @abc.abstractmethod
    def forward(
        self,
        static: pt.DataFrame[InferredSchema],
        dynamic: pt.DataFrame[InferredSchema],
        probabilistic: pt.DataFrame[InferredSchema],
    ) -> pt.DataFrame[InferredSchema]:
        ...


class IterativeResolution(ConflictResolution):
    """Process common symbols in each DataFrame individually"""

    @abc.abstractmethod
    def forward(
        self,
        static: pt.DataFrame[InferredSchema],
        dynamic: pt.DataFrame[InferredSchema],
        probabilistic: pt.DataFrame[InferredSchema],
        metadata: Metadata,
    ) -> pt.DataFrame[InferredSchema] | None:
        ...

    def _resolve(
        self,
        static: pt.DataFrame[InferredSchema],
        dynamic: pt.DataFrame[InferredSchema],
        probabilistic: pt.DataFrame[InferredSchema],
    ) -> pt.DataFrame[InferredSchema]:

        updates: list[pt.DataFrame[InferredSchema]] = []

        for (file, category, qname) in self.reference.itertuples(index=False):
            _static = static[
                (static["file"] == file)
                & (static["category"] == category)
                & (static["qname"] == qname)
            ]
            _dynamic = dynamic[
                (dynamic["file"] == file)
                & (dynamic["category"] == category)
                & (dynamic["qname"] == qname)
            ]
            _probabilistic = probabilistic[
                (probabilistic["file"] == file)
                & (probabilistic["category"] == category)
                & (probabilistic["qname"] == qname)
            ]
            _metadata = Metadata(file=file, category=category, qname=qname)

            update = self.forward(
                static=_static,
                dynamic=_dynamic,
                probabilistic=_probabilistic,
                metadata=_metadata,
            )

            if update is None:
                participants = "+".join(
                    filter(
                        None,
                        (
                            _static["method"].iloc[0] if not _static.empty else "",
                            _dynamic["method"].iloc[0] if not _dynamic.empty else "",
                            _probabilistic["method"].iloc[0] if not _probabilistic.empty else "",
                        ),
                    )
                )

                update = pt.DataFrame[InferredSchema](
                    {
                        "method": [participants],
                        "file": [_metadata.file],
                        "category": [_metadata.category],
                        "qname": [_metadata.qname],
                        "anno": [_ConflictResolution.UNRESOLVED],
                    }
                )

            assert len(update) == 1
            updates.append(update)

        return pd.concat(updates, ignore_index=True).pipe(pt.DataFrame[InferredSchema])
