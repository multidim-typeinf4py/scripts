import abc
import enum
import pathlib

from common.storage import MergedAnnotations

import pandas as pd


class Statistic(enum.Enum):
    COVERAGE = "coverage"
    ACCURACY = "accuracy"

    def __str__(self) -> str:
        return str(self.name)


class StatisticImpl:
    @property
    @abc.abstractmethod
    def ident(self) -> Statistic:
        ...

    @abc.abstractmethod
    def forward(self, *, repos: list[pathlib.Path], annotations: MergedAnnotations) -> pd.DataFrame:
        ...
