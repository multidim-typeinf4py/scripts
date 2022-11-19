import abc
import enum
import pathlib

from common.schemas import MergedAnnotationSchemaColumns, TypeCollectionSchema
from common.storage import MergedAnnotations

import pandas as pd
from pandas._libs import missing
import pandera.typing as pt


class Statistic(enum.Enum):
    COVERAGE = "coverage"

    def __str__(self) -> str:
        return str(self.name)


class StatisticOutput(dict[pathlib.Path, pd.DataFrame]):
    ...


class StatisticImpl:
    @property
    @abc.abstractmethod
    def ident(self) -> Statistic:
        ...

    @abc.abstractmethod
    def forward(
        self, *, repos: list[pathlib.Path], annotations: MergedAnnotations
    ) -> StatisticOutput:
        ...


class Coverage(StatisticImpl):
    ident = Statistic.COVERAGE

    def forward(
        self, *, repos: list[pathlib.Path], annotations: MergedAnnotations
    ) -> StatisticOutput:
        return StatisticOutput(
            {
                repo: annotations.df[f"{repo.name}_anno"].value_counts(normalize=True)[missing.NA]
                for repo in repos
            }
        )
