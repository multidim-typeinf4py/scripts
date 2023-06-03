import abc
import enum
import pathlib

from scripts.common import MergedAnnotations

import pandas as pd


class Statistic(enum.Enum):
    COVERAGE = "coverage"
    ACCURACY = "accuracy"

    def __str__(self) -> str:
        return str(self.name)


class StatisticImpl:
    def _read_anno_for_repo(self, repo: pathlib.Path, annotations: MergedAnnotations) -> pd.Series:
        return annotations.df[f"{repo.name}_anno"]

    @property
    @abc.abstractmethod
    def ident(self) -> Statistic:
        ...

    @abc.abstractmethod
    def forward(self, *, repos: list[pathlib.Path], annotations: MergedAnnotations) -> pd.DataFrame:
        ...
