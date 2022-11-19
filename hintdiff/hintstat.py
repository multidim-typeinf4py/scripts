import abc
import enum
import pathlib

from common.schemas import MergedAnnotationSchemaColumns, TypeCollectionSchema
from common.storage import MergedAnnotations

import pandas as pd
import pandera.typing as pt


class Statistic(enum.Enum):
    COVERAGE = "coverage"

    def __str__(self) -> str:
        return str(self.name)


class StatisticOutput(dict[pathlib.Path, pd.DataFrame]):
    ...


class StatisticImpl:
    def __init__(self, projects: list[pathlib.Path]) -> None:
        self._projects = projects

    @property
    @abc.abstractmethod
    def ident(self) -> Statistic:
        ...

    @abc.abstractmethod
    def forward(self, annotations: MergedAnnotations) -> StatisticOutput:
        ...


class Coverage(StatisticImpl):
    ident = Statistic.COVERAGE

    def forward(
        self, *, repos: list[pathlib.Path], annotations: MergedAnnotations
    ) -> StatisticOutput:
        return StatisticOutput(
            {
                repo: annotations.df[f"{repo.name}_anno"]
                .notna()
                .value_counts(normalize=True)[True]
                for repo in repos
            }
        )
