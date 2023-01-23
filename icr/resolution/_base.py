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
    qname_ssa: str


class ConflictResolution(abc.ABC):
    UNRESOLVED = missing.NA

    @property
    @abc.abstractmethod
    def method(self) -> str:
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
        common_cols = [InferredSchema.file, InferredSchema.category, InferredSchema.qname_ssa]

        static_safe: pt.DataFrame[InferredSchema] = pd.merge(
            left=static, right=self.reference, how=how, on=common_cols
        ).pipe(pt.DataFrame[InferredSchema])
        dynamic_safe: pt.DataFrame[InferredSchema] = pd.merge(
            left=dynamic, right=self.reference, how=how, on=common_cols
        ).pipe(pt.DataFrame[InferredSchema])
        probabilistic_safe: pt.DataFrame[InferredSchema] = pd.merge(
            left=probabilistic, right=self.reference, how=how, on=common_cols
        ).pipe(pt.DataFrame[InferredSchema])

        inferred = self._resolve(static_safe, dynamic_safe, probabilistic_safe)

        # Readd symbols with unresolved that were removed due to no
        # tool making a prediction
        method_names = [
            inf[InferredSchema.method].iloc[0]
            for inf in [static_safe, dynamic_safe, probabilistic_safe]
            if len(inf)
        ]
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
        # TODO: Make this a ResolvedSchema, as TopN does not apply anymore
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

        for (file, category, qname, qname_ssa) in self.reference.itertuples(index=False):
            _static = static[
                (static[InferredSchema.file] == file)
                & (static[InferredSchema.category] == category)
                & (static[InferredSchema.qname_ssa] == qname_ssa)
            ]
            _dynamic = dynamic[
                (dynamic[InferredSchema.file] == file)
                & (dynamic[InferredSchema.category] == category)
                & (dynamic[InferredSchema.qname_ssa] == qname_ssa)
            ]
            _probabilistic = probabilistic[
                (probabilistic[InferredSchema.file] == file)
                & (probabilistic[InferredSchema.category] == category)
                & (probabilistic[InferredSchema.qname_ssa] == qname_ssa)
            ]
            _metadata = Metadata(file=file, category=category, qname=qname, qname_ssa=qname_ssa)

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
                        "anno": [ConflictResolution.UNRESOLVED],
                    }
                )

            assert len(update) == 1
            updates.append(update)

        return pd.concat(updates, ignore_index=True).pipe(pt.DataFrame[InferredSchema])
